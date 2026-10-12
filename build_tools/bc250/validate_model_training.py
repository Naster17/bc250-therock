# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Short real-model full fine-tuning, AMP LoRA and fresh-process resume gates."""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformer_logging
from huggingface_hub.utils import disable_progress_bars


TEXT = (
    "The AMD BC-250 uses shared memory for CPU and GPU work. "
    "A training checkpoint stores model weights, optimizer state and random state. "
    "Native GPU execution must produce correct results."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("full", "lora-fp16", "lora-bf16"), required=True
    )
    parser.add_argument("--phase", choices=("train", "resume"), required=True)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    disable_progress_bars()
    transformer_logging.disable_progress_bar()
    if not 1 <= args.steps <= 8:
        parser.error("Use one to eight steps for this bounded qualification")
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    torch.set_num_threads(8)
    torch.manual_seed(1704)
    torch.cuda.manual_seed_all(1704)
    directory = args.directory / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "training-state.pt"
    dtype = {
        "full": torch.float32,
        "lora-fp16": torch.float16,
        "lora-bf16": torch.bfloat16,
    }[args.mode]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=32)
    labels = inputs["input_ids"].clone()
    gpu_inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
    report = {
        "status": "running",
        "mode": args.mode,
        "phase": args.phase,
        "model": json.loads((args.model / "bc250-model-provenance.json").read_text()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate")
        },
        "checks": [],
        "losses": [],
    }
    output = directory / f"{args.phase}.json"

    def passed(name: str) -> None:
        torch.cuda.synchronize()
        report["checks"].append(name)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"MODEL_TRAINING_PASS {args.mode} {args.phase} {name}", flush=True)

    def base(path: Path) -> torch.nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
            use_cache=False,
        )

    model = base(
        directory / "model"
        if args.mode == "full" and args.phase == "resume"
        else args.model
    )
    if args.mode != "full":
        if args.phase == "resume":
            model = PeftModel.from_pretrained(
                model, directory / "adapter", is_trainable=True
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0,
                    target_modules=["q_proj", "v_proj"],
                    bias="none",
                ),
            )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.cuda()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    report["trainable_parameters"] = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=2e-5 if args.mode == "full" else 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=dtype == torch.float16, init_scale=128
    )
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
        and (
            "layers.0.self_attn.q_proj.base_layer.weight" in name
            or "layers.0.self_attn.v_proj.base_layer.weight" in name
        )
    }
    state = None
    if args.phase == "resume":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["gpu_rng"])
        torch.testing.assert_close(
            torch.rand(17), state["cpu_rng_probe"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.rand(17, device="cuda").cpu(), state["gpu_rng_probe"], rtol=0, atol=0
        )
        model.eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            logits = model(**gpu_inputs).logits.cpu()
        torch.testing.assert_close(logits, state["logits"], rtol=5e-4, atol=3e-4)
        passed("fresh_container_model_optimizer_scheduler_scaler_rng_restore")
    elif args.mode == "full":
        # Independent CPU forward and gradients before changing pretrained weights.
        cpu = base(args.model).eval()
        with sdpa_kernel(SDPBackend.MATH):
            reference = cpu(**inputs, labels=labels)
            reference.loss.backward()
            actual = model(**gpu_inputs, labels=labels.cuda())
            actual.loss.backward()
        observed_logits = actual.logits.detach().cpu()
        reference_logits = reference.logits.detach()
        # Deep FP32 accumulation amplifies local rounding around zero. Bound
        # aggregate and maximum error as well as the elementwise comparison.
        report["cpu_logit_relative_l2"] = (
            (observed_logits - reference_logits).norm() / reference_logits.norm()
        ).item()
        report["cpu_logit_max_abs"] = (
            (observed_logits - reference_logits).abs().max().item()
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        assert report["cpu_logit_relative_l2"] < 2e-5
        assert report["cpu_logit_max_abs"] < 2e-3
        torch.testing.assert_close(
            observed_logits, reference_logits, rtol=5e-4, atol=1e-3
        )
        torch.testing.assert_close(
            actual.loss.cpu(), reference.loss.detach(), rtol=1e-4, atol=1e-4
        )
        cpu_parameters = dict(cpu.named_parameters())
        report["cpu_gradient_relative_l2"] = {}
        for name in (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
        ):
            expected = cpu_parameters[name].grad
            observed = dict(model.named_parameters())[name].grad.cpu()
            relative = (observed - expected).norm() / expected.norm().clamp_min(1e-12)
            report["cpu_gradient_relative_l2"][name] = relative.item()
            output.write_text(json.dumps(report, indent=2) + "\n")
            assert relative.item() < 2e-3, (name, relative.item())
            # Bound the worst component too, with scale-aware headroom for
            # near-zero cancellation propagated through the 30-layer model.
            max_error = (observed - expected).abs().max().item()
            assert max_error < 1e-4 + 1e-3 * expected.abs().max().item(), (
                name,
                max_error,
            )
        del (
            reference,
            actual,
            cpu,
            cpu_parameters,
            expected,
            observed,
            observed_logits,
            reference_logits,
        )
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        passed("cpu_reference_logits_loss_and_four_parameter_gradients")
    model.train()
    before = (
        [parameter.detach().clone() for parameter in trainable]
        if args.mode != "full"
        else [trainable[0].detach().clone()]
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        initial_loss = model(**gpu_inputs, labels=labels.cuda()).loss.item()
    steps = 1 if args.phase == "resume" else args.steps
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=dtype, enabled=dtype != torch.float32
        ), sdpa_kernel(SDPBackend.MATH):
            result = model(**gpu_inputs, labels=labels.cuda())
            loss = result.loss
        assert torch.isfinite(loss).item()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert any(
            parameter.grad is not None and parameter.grad.abs().max().item() > 0
            for parameter in trainable
        )
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in trainable
        )
        torch.nn.utils.clip_grad_norm_(trainable, 1)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert (
            scaler.get_scale() >= scale_before
        ), "Unexpected AMP overflow/skipped optimizer step"
        scheduler.step()
        report["losses"].append(loss.item())
        print(f"MODEL_STEP {args.mode} {step + 1} loss={loss.item():.6f}", flush=True)
    model.eval()
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        final = model(**gpu_inputs, labels=labels.cuda())
        final_loss, logits = final.loss.item(), final.logits.cpu()
    assert final_loss < initial_loss, (initial_loss, final_loss)
    assert any(
        not torch.equal(parameter.detach(), original)
        for parameter, original in zip(trainable, before)
    )
    for name, value in frozen.items():
        torch.testing.assert_close(
            dict(model.named_parameters())[name], value, rtol=0, atol=0
        )
    if args.mode != "full":
        assert frozen and all(
            "lora_" in name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    report.update(
        initial_loss=initial_loss,
        final_loss=final_loss,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        resumed_steps=(state["steps"] if state else 0) + steps,
    )
    passed("finite_gradients_parameter_updates_decreasing_loss_and_frozen_base")
    if args.phase == "train":
        if args.mode != "full":
            model.peft_config["default"].base_model_name_or_path = report["model"][
                "model"
            ]
            model.peft_config["default"].revision = report["model"]["revision"]
        model.save_pretrained(
            directory / ("model" if args.mode == "full" else "adapter"),
            safe_serialization=True,
        )
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        training_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": cpu_rng,
            "gpu_rng": gpu_rng,
            "cpu_rng_probe": torch.rand(17),
            "gpu_rng_probe": torch.rand(17, device="cuda").cpu(),
            "logits": logits,
            "steps": steps,
        }
        torch.save(training_state, checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)
        passed("persistent_model_or_adapter_and_complete_training_checkpoint")
    else:
        assert all(
            item["step"].item() == report["resumed_steps"]
            for item in optimizer.state.values()
        )
        passed("optimizer_step_continues_after_resume")
    report["status"] = "passed"
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
