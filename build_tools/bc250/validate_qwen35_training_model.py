#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic-only Qwen3.5 LoRA/checkpoint integration; never opens user data/checkpoints."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path("/home/nik/finetune-n7")
sys.path.insert(0, str(PROJECT / "runtime/transformers-5.19.0-cp312"))
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
import qwen35_training_ops as ops


def reference_conv(x, weight, bias=None, activation=None, **kwargs):
    """Independent ATen GPU tap sum, checked separately against CPU float64 conv."""
    assert x.is_cuda and not weight.requires_grad
    k = weight.shape[-1]
    padded = F.pad(x.float(), (k - 1, 0))
    y = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(k):
        y = (
            y
            + padded[..., tap : tap + x.shape[-1]]
            * weight[:, tap].float()[None, :, None]
        )
    if bias is not None:
        y = y + bias.float()[None, :, None]
    y = y.to(x.dtype)
    return F.silu(y) if activation in ("silu", "swish") else y


def make_model(actual, dtype):
    if actual:
        path = PROJECT / "models/Qwen3.5-2B-Base"
        config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
        model, info = Qwen3_5ForCausalLM.from_pretrained(
            path,
            config=config,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
            key_mapping={r"^model.language_model\.": "model."},
            output_loading_info=True,
        )
        assert not any(
            info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
        ), info
    else:
        config = Qwen3_5TextConfig(
            vocab_size=257,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=2048,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            },
        )
        config._attn_implementation = "sdpa"
        model = Qwen3_5ForCausalLM(config).to(dtype)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".layers." in name
    ]
    assert targets
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16 if actual else 4,
            lora_alpha=32 if actual else 8,
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    ).cuda()
    model.enable_input_require_grads()
    # Nonzero adapter B exercises both A and B gradients; synthetic test weights only.
    if not actual:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.01)
    return model


def run(model, ids):
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with sdpa_kernel(SDPBackend.MATH):
        output = model(input_ids=ids, labels=ids, use_cache=False)
        loss = output.loss
        del output
        loss.backward()
    torch.cuda.synchronize()
    gradients = {
        name: p.grad.detach().cpu().float().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients.values())
    return (
        dict(
            loss=float(loss.detach()),
            seconds=time.monotonic() - start,
            peak_allocated=torch.cuda.max_memory_allocated(),
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
        ),
        gradients,
    )


def pair(model, checkpoint, length):
    if checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()
    ids = torch.randint(3, model.config.vocab_size, (1, length), device="cuda")
    qwen.causal_conv1d_fn = reference_conv
    original_loss = model.get_base_model().loss_function
    baseline, ref = run(model, ids)
    ops.install(model)
    candidate, grad = run(model, ids)
    assert abs(baseline["loss"] - candidate["loss"]) < 0.015, (baseline, candidate)
    assert grad.keys() == ref.keys()
    errors = {}
    for name, g in grad.items():
        error = float(
            torch.linalg.vector_norm(g - ref[name])
            / torch.linalg.vector_norm(ref[name]).clamp_min(1e-12)
        )
        assert error < 0.08, (name, error)
        errors[name] = error
    # Keep global hook ownership clean for the next model's independently selected reference.
    qwen.causal_conv1d_fn = reference_conv
    if hasattr(qwen, "_bc250_original_causal_conv"):
        del qwen._bc250_original_causal_conv
    model.get_base_model().loss_function = original_loss
    return dict(
        checkpoint=checkpoint,
        length=length,
        reference=baseline,
        native=candidate,
        gradient_count=len(grad),
        max_adapter_gradient_relative_l2=max(errors.values()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--actual", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(483)
    torch.set_num_threads(4)
    result = dict(
        actual=args.actual,
        source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
        checks=[],
        reference="independent ATen GPU convolution tap sum and original Transformers loss",
    )
    try:
        for dtype in (
            (torch.bfloat16,) if args.actual else (torch.float32, torch.bfloat16)
        ):
            model = make_model(args.actual, dtype)
            for checkpoint in ((True,) if args.actual else (False, True)):
                row = pair(model, checkpoint, 65)
                row["dtype"] = str(dtype)
                result["checks"].append(row)
            if args.actual:
                ops.install(model)
                ids = torch.randint(
                    3, model.config.vocab_size, (1, 1024), device="cuda"
                )
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=1e-4
                )
                first = next(
                    p for name, p in model.named_parameters() if "lora_B" in name
                )
                before = first.detach().clone()
                row, grad = run(model, ids)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                assert not torch.equal(first, before)
                row.update(
                    length=1024,
                    checkpoint=True,
                    adapter_optimizer_update=True,
                    gradient_count=len(grad),
                )
                result["checks"].append(row)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result["passed"] = True
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
