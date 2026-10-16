#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-checked paired eager training timings; synthetic tensors only."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch


def measure(function):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall = time.perf_counter()
    start.record()
    for _ in range(5):
        function()
    end.record()
    end.synchronize()
    gpu_us = start.elapsed_time(end) * 200
    wall_us = (time.perf_counter() - wall) * 2e5
    assert gpu_us > 0 and wall_us > 0
    return gpu_us, wall_us


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    torch.manual_seed(251018)
    torch.set_num_threads(2)
    rows = []
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for m, n, k in [(32, 16, 32), (128, 64, 128)]:
            for train_weight in [False, True]:
                group = 2 if dtype == torch.float32 else 4
                mask = torch.zeros(m, k // group, group, device="cuda", dtype=torch.bool)
                mask[:, :, :group // 2] = True
                mask = mask.view(m, k)
                a = (torch.randn(m, k, device="cuda", dtype=dtype) * .05 * mask).requires_grad_(train_weight)
                b = (torch.randn(k, n, device="cuda", dtype=dtype) * .05).requires_grad_()
                upstream = torch.randn(m, n, device="cuda", dtype=dtype) * .1
                packed = torch._cslt_compress(a.detach())
                inputs = [a, b] if train_weight else [b]
                def sparse():
                    p = torch._cslt_compress(a) if train_weight else packed
                    y = torch._cslt_sparse_mm(p, b)
                    return torch.autograd.grad(y, inputs, upstream)
                def dense():
                    y = ((a * mask) if train_weight else a) @ b
                    return torch.autograd.grad(y, inputs, upstream)
                ca = a.detach().cpu().double().requires_grad_(train_weight)
                cb = b.detach().cpu().double().requires_grad_()
                y = (ca * mask.cpu()) @ cb
                refs = torch.autograd.grad(y, [ca, cb] if train_weight else [cb], upstream.cpu().double())
                tolerance = .003 if dtype == torch.bfloat16 else .0003 if dtype == torch.float16 else 2e-6
                for function in [sparse, dense]:
                    for value, reference in zip(function(), refs):
                        torch.testing.assert_close(value.cpu().double(), reference, atol=tolerance, rtol=.04 if dtype != torch.float32 else 1e-4)
                for _ in range(2):
                    sparse(); dense()
                samples = {"sparse": [], "dense": []}
                for pair in range(6):
                    order = [("sparse", sparse), ("dense", dense)]
                    if pair % 2:
                        order.reverse()
                    for name, function in order:
                        samples[name].append(measure(function))
                medians = {name: dict(gpu_us=statistics.median(v[0] for v in values), wall_us=statistics.median(v[1] for v in values)) for name, values in samples.items()}
                rows.append(dict(status="passed", dtype=str(dtype), shape=[m, n, k], train_weight=train_weight, samples=samples, medians=medians, dense_over_sparse=medians["dense"]["gpu_us"] / medians["sparse"]["gpu_us"]))
    result = dict(status="passed", results=rows, scope="Eager forward+autograd.grad; six alternating rounds x5 calls. Frozen weight reuses pack; trainable weight recompresses, dense reference applies same mask. Includes dense GPU backward intermediates and host synchronization; no application/kernel-only speed claim.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
