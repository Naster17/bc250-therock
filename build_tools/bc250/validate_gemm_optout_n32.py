#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the explicit opt-out retains the old default solution path."""
import argparse
import json
import os
from pathlib import Path
import torch
from benchmark_gemm_n32 import check_reference, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("BC250_GEMM_TUNING") == "0"
    assert "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH" not in os.environ
    torch.set_num_threads(4)
    fixture = args.output.parent.parent / "gemm-n32-layout-r1/fixture-9db073af72b1.pt"
    xs = torch.load(fixture, weights_only=True)
    a, b = xs["a"].cuda(), xs["b"].cuda()
    expected = xs["a"].double() @ xs["b"].double()
    numerical = check_reference(a @ b, expected)
    timing = measure(lambda: torch.mm(a, b), 3)
    assert timing["gpu_ms"] > 50, timing
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                checks=["explicit-optout", "cpu-reference-original-path"],
                numerical=numerical,
                timing=timing,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
