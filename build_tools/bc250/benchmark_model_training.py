# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded steady training-step timings for the qualified pretrained model."""

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformer_logging
from peft import PeftModel
from huggingface_hub.utils import disable_progress_bars


TEXT = (
    "The AMD BC-250 uses shared memory for CPU and GPU work. "
    "A training checkpoint stores model weights, optimizer state and random state. "
    "Native GPU execution must produce correct results."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="measure LoRA without activation recomputation",
    )
    args = parser.parse_args()
    disable_progress_bars()
    transformer_logging.disable_progress_bar()
    torch.set_num_threads(8)
    torch.manual_seed(1704)
    device = torch.cuda.get_device_properties(0)
    assert device.gcnArchName.split(":")[0] == "gfx1013"
    tokenizer = AutoTokenizer.from_pretrained(
        args.workspace / "training-model", local_files_only=True
    )
    inputs = {
        name: tensor.cuda()
        for name, tensor in tokenizer(
            TEXT, return_tensors="pt", truncation=True, max_length=32
        ).items()
    }
    labels = inputs["input_ids"].clone()
    report = {
        "status": "running",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "arch": device.gcnArchName,
        "batch": 1,
        "sequence_length": labels.shape[1],
        "optimizer": "AdamW",
        "lr": 0,
        "notes": "lr=0 holds weights fixed during timing; optimizer state updates still execute",
        "cases": [],
    }
    modes = (
        (
            ("lora-fp32", torch.float32),
            ("lora-fp16", torch.float16),
            ("lora-bf16", torch.bfloat16),
        )
        if args.no_checkpoint
        else (
            ("full", torch.float32),
            ("lora-fp16", torch.float16),
            ("lora-bf16", torch.bfloat16),
        )
    )
    for mode, dtype in modes:
        path = args.workspace / (
            "training-validation/full/model" if mode == "full" else "training-model"
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.float32,
            attn_implementation="sdpa",
            use_cache=False,
            local_files_only=True,
        )
        if mode != "full":
            adapter = "lora-fp16" if args.no_checkpoint else mode
            model = PeftModel.from_pretrained(
                model,
                args.workspace / "training-validation" / adapter / "adapter",
                is_trainable=True,
            )
            if not args.no_checkpoint:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        assert model.is_gradient_checkpointing == (
            mode != "full" and not args.no_checkpoint
        )
        model.cuda().train()
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(parameters, lr=0)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=dtype == torch.float16, init_scale=128
        )

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=dtype, enabled=dtype != torch.float32
            ), sdpa_kernel(SDPBackend.MATH):
                loss = model(**inputs, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            scaler.step(optimizer)
            scaler.update()
            return loss

        first_started = time.perf_counter()
        first_loss = step()
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - first_started) * 1000
        assert torch.isfinite(first_loss).item()
        initial_scale = scaler.get_scale()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(3):
                loss = step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000 / 3)
        assert torch.isfinite(loss).item()
        assert (
            scaler.get_scale() >= initial_scale
        ), "AMP skipped an update due to overflow"
        torch.testing.assert_close(loss, first_loss, rtol=1e-4, atol=1e-4)
        median = statistics.median(samples)
        quartiles = statistics.quantiles(samples, n=4)
        case = {
            "name": mode,
            "step_samples_ms": samples,
            "step_median_ms": median,
            "step_iqr_percent": (quartiles[2] - quartiles[0]) / median * 100,
            "tokens_per_second": labels.numel() * 1000 / median,
            "first_step_ms": first_ms,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "activation_checkpointing": model.is_gradient_checkpointing,
        }
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"MODEL_PERF {mode} step_ms={median:.3f} tokens_s={case['tokens_per_second']:.1f} iqr_pct={case['step_iqr_percent']:.2f}",
            flush=True,
        )
        del model, optimizer, scaler, parameters, first_loss, loss
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("MODEL_PERFORMANCE_SUITE_PASS", flush=True)


if __name__ == "__main__":
    main()
