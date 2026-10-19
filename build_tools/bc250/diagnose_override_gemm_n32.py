#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check whether ordinary solution enumeration hydrates a cold override."""
import argparse
import json
from pathlib import Path
import torch
from benchmark_gemm_n32 import Gemm, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.output.parent.parent / "gemm-n32-layout-r1/fixture-9db073af72b1.pt"
    xs = torch.load(path, weights_only=True)
    a, b = xs["a"].cuda(), xs["b"].cuda()
    native = Gemm(a, b)
    try:
        available = native.solutions()
        native.call(0)
        out = a @ b
        torch.testing.assert_close(out, native.out, atol=0, rtol=0)
        timing = measure(lambda: torch.mm(a, b), 3)
        assert timing["gpu_ms"] < 20, timing
        args.output.write_text(
            json.dumps(
                dict(
                    status="passed",
                    checks=["enumeration-hydrates-override"],
                    available=available,
                    timing=timing,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        native.close()


if __name__ == "__main__":
    main()
