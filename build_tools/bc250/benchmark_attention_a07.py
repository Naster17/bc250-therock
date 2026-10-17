#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paired eager head256 forward/backward event/wall/peak measurements."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--candidate", type=Path)
    args = ap.parse_args()
    if args.candidate:
        spec = importlib.util.spec_from_file_location(
            "attention_benchmark_candidate", args.candidate
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        from torch.nn.attention import _gfx1013 as mod
    torch.manual_seed(71707)
    torch.set_num_threads(4)
    result = dict(
        status="running",
        results=[],
        source_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        protocol="Six alternating pairs after two warmups/path. Eager forward+autograd.grad, HIP-event/wall envelopes and peak allocated minus persistent inputs. No graphs/kernel-only/application gain claim.",
    )
    try:
        for dtype, length, h, hk in [
            (torch.float32, 256, 4, 2),
            (torch.bfloat16, 256, 8, 2),
            (torch.bfloat16, 1024, 8, 2),
            (torch.bfloat16, 2048, 8, 2),
            (torch.bfloat16, 4096, 2, 1),
        ]:
            xs = [
                (
                    torch.randn(1, heads, length, 256, device="cuda", dtype=dtype) * 0.2
                ).requires_grad_()
                for heads in (h, hk, hk)
            ]
            dy = torch.randn_like(xs[0]) * 0.1

            def native():
                return mod.scaled_dot_product_attention(
                    *xs, is_causal=True, enable_gqa=True
                )

            def math():
                with sdpa_kernel(SDPBackend.MATH):
                    return F.scaled_dot_product_attention(
                        *xs, is_causal=True, enable_gqa=True
                    )

            yn, yr = native(), math()
            gn, gr = torch.autograd.grad(yn, xs, dy), torch.autograd.grad(yr, xs, dy)
            errors = [
                float(
                    (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)
                )
                for a, b in [(yn, yr), *zip(gn, gr)]
            ]
            assert max(errors) < (0.006 if dtype == torch.bfloat16 else 5e-6), errors
            del yn, yr, gn, gr

            def train(fn):
                return torch.autograd.grad(fn(), xs, dy)

            for fn in (math, native):
                for _ in range(2):
                    train(fn)
            samples = {name: [] for name in ["math", "tiled"]}
            for pair in range(6):
                order = [("math", math), ("tiled", native)]
                if pair % 2:
                    order.reverse()
                for name, fn in order:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    wall = time.perf_counter()
                    start.record()
                    grads = train(fn)
                    end.record()
                    end.synchronize()
                    samples[name].append(
                        dict(
                            gpu_ms=start.elapsed_time(end),
                            wall_ms=(time.perf_counter() - wall) * 1000,
                            extra_peak_bytes=torch.cuda.max_memory_allocated()
                            - allocated,
                        )
                    )
                    del grads
            medians = {
                name: {key: statistics.median(x[key] for x in data) for key in data[0]}
                for name, data in samples.items()
            }
            result["results"].append(
                dict(
                    status="passed",
                    dtype=str(dtype),
                    length=length,
                    h=h,
                    hk=hk,
                    d=256,
                    output_qkv_relative_l2=errors,
                    medians=medians,
                    samples=samples,
                    math_over_tiled=medians["math"]["gpu_ms"]
                    / medians["tiled"]["gpu_ms"],
                )
            )
            print(dtype, length, medians, flush=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        result["status"] = "passed"
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
