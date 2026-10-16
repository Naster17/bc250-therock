#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic LoRA references/timings; never opens user datasets/checkpoints."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

import test_bc250_gated_delta as fixtures
import validate_qwen35_training_model as existing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--actual", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(61710)
    torch.set_num_threads(4)
    ops = fixtures.ops
    original = existing.qwen.torch_chunk_gated_delta_rule

    def candidate(
        *values,
        cu_seqlens=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs
    ):
        assert (
            cu_seqlens is None and not use_cache and not output_attentions
        ), "Packed sequences/cache are not qualified"
        return ops.chunk_gated_delta_rule(*values, **kwargs)

    report = dict(
        status="running",
        actual=args.actual,
        results=[],
        source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
        protocol="Same model/input/helper/math SDPA; 3 alternating pairs after both-path warmup. Times include forward/backward, exclude CPU gradient copies. Synthetic data only.",
    )
    try:
        for dtype in (
            (torch.bfloat16,) if args.actual else (torch.float32, torch.bfloat16)
        ):
            model = existing.make_model(args.actual, dtype)
            existing.ops.install(model)
            for checkpoint in ((True,) if args.actual else (False, True)):
                if checkpoint:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                else:
                    model.gradient_checkpointing_disable()
                for length in ((65, 1024) if args.actual else (65,)):
                    ids = torch.randint(
                        3, model.config.vocab_size, (1, length), device="cuda"
                    )

                    def run(fn):
                        existing.qwen.torch_chunk_gated_delta_rule = fn
                        model.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        allocated = torch.cuda.memory_allocated()
                        begin = time.perf_counter()
                        with sdpa_kernel(SDPBackend.MATH):
                            loss = model(
                                input_ids=ids, labels=ids, use_cache=False
                            ).loss
                            loss.backward()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - begin
                        data = dict(
                            seconds=elapsed,
                            loss=float(loss.detach()),
                            peak_allocated=torch.cuda.max_memory_allocated(),
                            extra_peak_bytes=torch.cuda.max_memory_allocated()
                            - allocated,
                        )
                        grads = {
                            name: p.grad.detach().cpu().float().clone()
                            for name, p in model.named_parameters()
                            if p.requires_grad and p.grad is not None
                        }
                        assert grads and all(
                            bool(g.isfinite().all()) for g in grads.values()
                        )
                        return data, grads

                    _, ref = run(original)
                    _, actual = run(candidate)
                    assert actual.keys() == ref.keys()
                    errors = {
                        name: float(
                            torch.linalg.vector_norm(actual[name] - r)
                            / torch.linalg.vector_norm(r).clamp_min(1e-12)
                        )
                        for name, r in ref.items()
                    }
                    assert max(errors.values()) < (
                        0.002 if dtype == torch.float32 else 0.04
                    ), max(errors.values())
                    del actual, ref
                    samples = {"reference": [], "native": []}
                    for pair in range(3):
                        order = [("reference", original), ("native", candidate)]
                        if pair % 2:
                            order.reverse()
                        for name, fn in order:
                            data, gradients = run(fn)
                            del gradients
                            samples[name].append(data)
                    assert (
                        max(
                            abs(a["loss"] - b["loss"])
                            for a, b in zip(samples["reference"], samples["native"])
                        )
                        < 0.015
                    )
                    medians = {
                        name: {
                            key: statistics.median(d[key] for d in values)
                            for key in values[0]
                        }
                        for name, values in samples.items()
                    }
                    report["results"].append(
                        dict(
                            dtype=str(dtype),
                            length=length,
                            checkpoint=checkpoint,
                            adapter_gradient_count=len(errors),
                            max_adapter_gradient_relative_l2=max(errors.values()),
                            samples=samples,
                            medians=medians,
                            speed_ratio=medians["reference"]["seconds"]
                            / medians["native"]["seconds"],
                        )
                    )
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "passed"
    finally:
        existing.qwen.torch_chunk_gated_delta_rule = original
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
