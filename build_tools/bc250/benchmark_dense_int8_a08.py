#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded eager INT8 timings against resident FP32 GEMM plus integer copy."""
import argparse
import json
from pathlib import Path
import statistics
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1809)
    rows = []
    for m, n, k in [(32, 32, 32), (128, 128, 128), (256, 256, 256)]:
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 128, (k, n), dtype=torch.int8, device="cuda")
        fa, fb = a.float(), b.float()
        fo = torch.empty((m, n), device="cuda")
        io = torch.empty((m, n), device="cuda", dtype=torch.int32)
        fallback = torch.empty_like(io)
        reference = (a.cpu().long() @ b.cpu().long()).int()

        def native():
            torch._int_mm(a, b, out=io)

        def floating():
            torch.mm(fa, fb, out=fo)
            fallback.copy_(fo)

        for _ in range(2):
            native()
            floating()
        torch.cuda.synchronize()
        assert torch.equal(io.cpu(), reference) and torch.equal(
            fallback.cpu(), reference
        )
        samples = {"int8_ms": [], "fp32_plus_cast_ms": []}
        for pair in range(5):
            for name, fn in (
                [("int8_ms", native), ("fp32_plus_cast_ms", floating)]
                if pair % 2 == 0
                else [("fp32_plus_cast_ms", floating), ("int8_ms", native)]
            ):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(10):
                    fn()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 10)
        rows.append(
            dict(
                status="passed",
                m=m,
                n=n,
                k=k,
                **samples,
                median_int8_ms=statistics.median(samples["int8_ms"]),
                median_fp32_plus_cast_ms=statistics.median(samples["fp32_plus_cast_ms"])
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                results=rows,
                scope="Five alternating pairs, ten eager calls/event envelope after two warmups; resident inputs/preallocated outputs. FP32 baseline has an extra conversion kernel and resident float inputs; bounded exact small sums only. No kernel-only, quantization recipe, LoRA or universal speedup claim.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
