#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fresh-process public Torch proof of the exact-layout rocBLAS override."""
import argparse
import json
import os
from pathlib import Path

import torch
from benchmark_gemm_n32 import Gemm, check_reference, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH")
    torch.set_num_threads(4)
    source = args.output.parent.parent / "gemm-n32-layout-r1"
    report = dict(status="running", checks=[], results=[])
    try:
        for row in json.loads((source / "result.json").read_text())["layouts"]:
            xs = torch.load(source / row["fixture"], weights_only=True)
            a, b = xs["a"].cuda(), xs["b"].cuda()
            ref = xs["a"].double() @ xs["b"].double()
            out = a @ b
            numeric = check_reference(out, ref)
            native = Gemm(a, b)
            try:
                native.call(0)
                torch.testing.assert_close(native.out, out, atol=0, rtol=0)
                public = measure(lambda: torch.mm(a, b), 5)
                if row["a_stride"] == [1, 1024]:
                    native.call(-490)
                    torch.testing.assert_close(native.out, out, atol=0, rtol=0)
                    assert public["gpu_ms"] < 20, public
                native.guards()
            finally:
                native.close()
            report["results"].append(
                dict(layout=row, numerical=numeric, public_timing=public)
            )
            report["checks"].append(row["fixture"])
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
