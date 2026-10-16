#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile native preparation and complete Qwen3.5 gated-delta GPU operations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

PROJECT = Path("/home/nik/finetune-n7")
sys.path.insert(0, str(PROJECT / "runtime/transformers-5.19.0-cp312"))
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
import test_bc250_gated_delta as fixtures


def measure(fn, repeats: int) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    begin = time.perf_counter()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return dict(
        gpu_ms=start.elapsed_time(end) / repeats,
        wall_ms=(time.perf_counter() - begin) * 1000 / repeats,
        extra_peak_bytes=torch.cuda.max_memory_allocated() - allocated,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(61017)
    torch.set_num_threads(4)
    ops = fixtures.ops
    result = dict(
        status="running",
        results=[],
        source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
        transformer_sha256=hashlib.sha256(Path(qwen.__file__).read_bytes()).hexdigest(),
        protocol="6 alternating paired eager rounds; stage x5, full x1; forward+autograd.grad; warmups2; includes host submission, no kernel-only/application speed claim",
    )
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for length in (65, 1024):
                xs = fixtures.inputs(1, length, 16, 128, 128, dtype, "cuda")
                fp = [x.float() for x in xs]
                calls = {
                    "reference_prepare": lambda: fixtures.reference_prepare(*fp, True),
                    "native_prepare": lambda: ops.prepare_gated_delta(
                        *xs, normalize=True
                    ),
                    "reference_full": lambda: qwen.torch_chunk_gated_delta_rule(
                        *xs, use_qk_l2norm_in_kernel=True
                    )[0],
                    "native_full": lambda: ops.chunk_gated_delta_rule(
                        *xs, use_qk_l2norm_in_kernel=True
                    )[0],
                }
                yref, yn = calls["reference_full"](), calls["native_full"]()
                gy = torch.randn_like(yn) * 0.1
                gr = torch.autograd.grad(yref, xs, gy)
                gn = torch.autograd.grad(yn, xs, gy)
                errors = [
                    float(
                        torch.linalg.vector_norm(a.float() - b.float())
                        / torch.linalg.vector_norm(a.float()).clamp_min(1e-12)
                    )
                    for a, b in zip(gr, gn)
                ]
                assert max(errors) < (0.001 if dtype == torch.float32 else 0.02), errors
                torch.testing.assert_close(
                    yn,
                    yref,
                    atol=2e-5 if dtype == torch.float32 else 0.001,
                    rtol=0.001 if dtype == torch.float32 else 0.02,
                )
                del yref, yn, gr, gn
                samples = {name: [] for name in calls}

                def train(name):
                    # Recreate FP32 casts inside the graph on each call.
                    y = (
                        fixtures.reference_prepare(*(x.float() for x in xs), True)
                        if name == "reference_prepare"
                        else calls[name]()
                    )
                    if isinstance(y, tuple):
                        return torch.autograd.grad(
                            y, xs, tuple(torch.ones_like(o) * 0.01 for o in y)
                        )
                    return torch.autograd.grad(y, xs, gy)

                for name in calls:
                    for _ in range(2):
                        train(name)
                for round_id in range(6):
                    for kind in ("prepare", "full"):
                        order = ["reference_" + kind, "native_" + kind]
                        if round_id % 2:
                            order.reverse()
                        for name in order:
                            samples[name].append(
                                measure(
                                    lambda: train(name), 5 if kind == "prepare" else 1
                                )
                            )
                medians = {
                    name: {
                        key: statistics.median(x[key] for x in values)
                        for key in values[0]
                    }
                    for name, values in samples.items()
                }
                result["results"].append(
                    dict(
                        dtype=str(dtype),
                        length=length,
                        heads=16,
                        dk=128,
                        dv=128,
                        qkv_g_beta_gradient_relative_l2=errors,
                        medians=medians,
                        samples=samples,
                        prepare_speed_ratio=medians["reference_prepare"]["gpu_ms"]
                        / medians["native_prepare"]["gpu_ms"],
                        full_speed_ratio=medians["reference_full"]["gpu_ms"]
                        / medians["native_full"]["gpu_ms"],
                    )
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        result["status"] = "passed"
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
