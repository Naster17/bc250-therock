#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded sparse backward references or paired eager forward timings."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    if not args.benchmark:
        rows_file = args.output.with_name("rows.json")
        code = subprocess.call([sys.executable, str(Path(__file__).with_name("test_bc250_sparse_autograd.py")), "-v"], env=dict(os.environ, BC250_SPARSE_ROWS=str(rows_file)))
        rows = json.loads(rows_file.read_text()) if rows_file.exists() else []
        result = dict(status="passed" if code == 0 and len(rows) == 38 else "failed", results=rows, returncode=code)
    else:
        torch.manual_seed(251017)
        torch.set_num_threads(2)
        rows = []
        assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
        def measure(fn):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall = time.perf_counter()
            start.record()
            for _ in range(10):
                fn()
            end.record(); end.synchronize()
            gpu_us, wall_us = start.elapsed_time(end) * 100, (time.perf_counter() - wall) * 1e5
            assert gpu_us > 0 and wall_us > 0
            return gpu_us, wall_us
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for m, n, k in [(32, 16, 32), (128, 64, 128), (256, 128, 256), (512, 64, 512)]:
                group = 2 if dtype == torch.float32 else 4
                a = torch.randn(m, k, device="cuda", dtype=dtype) * .1
                a.view(m, k // group, group)[:, :, group // 2:] = 0
                b = torch.randn(k, n, device="cuda", dtype=dtype) * .1
                packed = torch._cslt_compress(a)
                def sparse(): return torch._cslt_sparse_mm(packed, b)
                def dense(): return a @ b
                def compress(): return torch._cslt_compress(a)
                reference = (a.cpu().double() @ b.cpu().double()).to(dtype)
                tolerance = .003 if dtype == torch.bfloat16 else .0005 if dtype == torch.float16 else 3e-5
                torch.testing.assert_close(sparse().cpu(), reference, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(dense().cpu(), reference, atol=tolerance, rtol=tolerance)
                for _ in range(3): sparse(); dense(); compress()
                samples = {name: [] for name in ["sparse", "dense", "compress"]}
                for pair in range(6):
                    order = [("sparse", sparse), ("dense", dense)]
                    if pair % 2: order.reverse()
                    for name, fn in order + [("compress", compress)]:
                        samples[name].append(measure(fn))
                medians = {name: dict(gpu_us=statistics.median(v[0] for v in values), wall_us=statistics.median(v[1] for v in values)) for name, values in samples.items()}
                rows.append(dict(status="passed", dtype=str(dtype), shape=[m, n, k], samples=samples, medians=medians, dense_over_sparse=medians["dense"]["gpu_us"] / medians["sparse"]["gpu_us"]))
        result = dict(status="passed", results=rows, scope="Paired eager HIP-event/wall forward calls; six alternating rounds x10. Prepacked sparse, compression separately. No graph/kernel-only/application speed claim.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
