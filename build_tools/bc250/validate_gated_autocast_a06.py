#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Autocast full-operation references with FP32 inputs and both AMP types."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import bc250_gated_delta as ops

import validate_qwen35_training_model as existing
import test_bc250_gated_delta as fixtures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--candidate", type=Path)
    args = ap.parse_args()
    global ops
    if args.candidate:
        spec = importlib.util.spec_from_file_location(
            "a06_amp_candidate", args.candidate
        )
        ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ops)
    torch.manual_seed(71706)
    torch.set_num_threads(4)
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for normalize in (False, True):
            xs = fixtures.inputs(1, 65, 2, 32, 16, torch.float32, "cuda")
            with torch.autocast("cuda", dtype=dtype):
                y = ops.chunk_gated_delta_rule(*xs, use_qk_l2norm_in_kernel=normalize)[
                    0
                ]
                r = existing.qwen.torch_chunk_gated_delta_rule(
                    *xs, use_qk_l2norm_in_kernel=normalize
                )[0]
            gy = torch.randn_like(y) * 0.1
            a = torch.autograd.grad(y, xs, gy)
            b = torch.autograd.grad(r, xs, gy)
            # Reference-ordered preparation/scan must exactly match this fixture.
            torch.testing.assert_close(y, r, atol=0, rtol=0)
            for ga, gb in zip(a, b):
                torch.testing.assert_close(ga, gb, atol=0, rtol=0)
            rows.append(
                dict(
                    status="passed",
                    dtype=str(dtype),
                    normalize=normalize,
                    exact_output_and_five_gradients=True,
                )
            )
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                results=rows,
                source_sha256=hashlib.sha256(
                    Path(ops.__file__).read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
