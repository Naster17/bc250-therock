#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retained runtime CPU/gradient gates plus a small real Triton JIT launch."""

import argparse
import json
import hashlib
from pathlib import Path
import runpy

import torch
import triton
import triton.language as tl


@triton.jit
def vector_kernel(x, y, n: tl.constexpr, BLOCK: tl.constexpr, BIAS: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < n, other=0)
    tl.store(y + offsets, values * 2 + BIAS, offsets < n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    runpy.run_path(
        str(Path(__file__).with_name("validate_cleanup_runtime.py")),
        run_name="__main__",
    )
    report = json.loads(args.output.read_text())
    cpu = torch.arange(513, dtype=torch.float32) / 128
    x = cpu.cuda()
    y = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    bias = int.from_bytes(
        hashlib.sha256(str(args.output.parent).encode()).digest()[:2], "little"
    )
    with torch.cuda.stream(stream):
        vector_kernel[(triton.cdiv(x.numel(), 128),)](x, y, x.numel(), 128, bias)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(y.cpu(), cpu * 2 + bias, rtol=0, atol=0)
    import triton._C.libtriton as native

    report["checks"].append("triton-jit-nondefault-stream-exact-513")
    report["triton_unique_specialization_bias"] = bias
    report["triton_module"] = triton.__file__
    report["triton_native"] = native.__file__
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "STORAGE_RUNTIME_AND_TRITON_PASS",
        len(report["checks"]),
        report["triton_native"],
        flush=True,
    )


if __name__ == "__main__":
    main()
