#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Nonzero synthetic LoRA: reference, corrected gated delta, then D256 SDPA."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

import validate_qwen35_training_model as existing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--a06-candidate", type=Path)
    ap.add_argument("--a07-candidate", type=Path)
    args = ap.parse_args()
    from torch.nn import bc250_gated_delta as a06
    from torch.nn.attention import _gfx1013 as a07

    if args.a06_candidate:
        spec = importlib.util.spec_from_file_location(
            "a06_corrected", args.a06_candidate
        )
        a06 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(a06)
    if args.a07_candidate:
        spec = importlib.util.spec_from_file_location("a07_d256", args.a07_candidate)
        a07 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(a07)
    torch.set_num_threads(4)
    original_delta = existing.qwen.torch_chunk_gated_delta_rule
    tiled_calls = []
    tiled_impl = a07.scaled_dot_product_attention

    def counted(*values, **kwargs):
        assert values[0].is_cuda and values[0].shape[-1] == 256
        tiled_calls.append(list(values[0].shape))
        return tiled_impl(*values, **kwargs)

    a07.scaled_dot_product_attention = counted
    a07.register_transformers_attention()

    def corrected(
        *values,
        cu_seqlens=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs
    ):
        assert cu_seqlens is None and not use_cache
        return a06.chunk_gated_delta_rule(*values, **kwargs)

    def compare(actual, expected):
        assert actual.keys() == expected.keys() and len(actual) == 372
        assert all(
            bool(x.isfinite().all()) and float(x.norm()) > 0 for x in actual.values()
        )
        errors = {
            n: float((g - expected[n]).norm() / expected[n].norm().clamp_min(1e-12))
            for n, g in actual.items()
        }
        return dict(
            max_relative_l2=max(errors.values()),
            global_relative_l2=(
                sum(float((g - expected[n]).norm()) ** 2 for n, g in actual.items())
                / sum(float(x.norm()) ** 2 for x in expected.values())
            )
            ** 0.5,
            exact=all(torch.equal(actual[n], expected[n]) for n in actual),
            worst_parameter=max(errors, key=errors.get),
            nonzero_gradients=372,
        )

    report = dict(
        status="running",
        results=[],
        source_sha256={
            name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
            for name, mod in [("a06", a06), ("a07", a07)]
        },
    )
    try:
        for dtype, length in [
            (torch.float32, 65),
            (torch.bfloat16, 65),
            (torch.bfloat16, 1024),
        ]:
            torch.manual_seed(61718)
            model = existing.make_model(True, dtype).train()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_B" in name:
                        p.normal_(0, 0.001)
            existing.ops.install(model)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            assert model.training and model.is_gradient_checkpointing
            ids = torch.randint(3, model.config.vocab_size, (1, length), device="cuda")
            model.get_base_model().set_attn_implementation("sdpa")
            existing.qwen.torch_chunk_gated_delta_rule = original_delta
            baseline, reference = existing.run(model, ids)
            existing.qwen.torch_chunk_gated_delta_rule = corrected
            fixed, gradients = existing.run(model, ids)
            parity = compare(gradients, reference)
            del gradients, reference
            # Fixed A06 must pass the unchanged original model contract.
            _, math_gradients = existing.run(model, ids)
            model.get_base_model().set_attn_implementation("bc250_tiled")
            before = len(tiled_calls)
            tiled, gradients = existing.run(model, ids)
            attention = compare(gradients, math_gradients)
            del gradients, math_gradients
            calls = len(tiled_calls) - before
            row = dict(
                dtype=str(dtype),
                length=length,
                a06=parity,
                a07=attention,
                reference=baseline,
                corrected_math=fixed,
                corrected_tiled=tiled,
                tiled_d256_calls=calls,
                model_training=model.training,
                gradient_checkpointing=model.is_gradient_checkpointing,
            )
            report["results"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert calls >= 12, calls
            row["a06_passed"] = (
                parity["max_relative_l2"] < (0.002 if dtype == torch.float32 else 0.04)
                and abs(baseline["loss"] - fixed["loss"]) < 0.015
            )
            row["a07_passed"] = (
                attention["max_relative_l2"]
                < (0.002 if dtype == torch.float32 else 0.04)
                and abs(fixed["loss"] - tiled["loss"]) < 0.015
            )
            print(
                dtype,
                length,
                "A06",
                parity,
                "A07",
                attention,
                "D256 calls",
                calls,
                flush=True,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = (
            "passed"
            if all(x["a06_passed"] and x["a07_passed"] for x in report["results"])
            else "numerics_failed"
        )
    finally:
        a07.scaled_dot_product_attention = tiled_impl
        existing.qwen.torch_chunk_gated_delta_rule = original_delta
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed"


if __name__ == "__main__":
    main()
