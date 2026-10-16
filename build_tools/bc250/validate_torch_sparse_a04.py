#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run native Torch test runner contracts; retain baseline unavailability separately."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    assert torch.cuda.get_device_properties(0).warp_size == 32
    if args.baseline:
        try:
            torch._cslt_compress(torch.ones(32, 32, device="cuda", dtype=torch.float16))
        except RuntimeError as e:
            assert "not supported" in str(e)
            args.output.write_text(
                json.dumps(
                    dict(
                        status="expected_unavailable",
                        error=str(e),
                        reported_available=torch.backends.cusparselt.is_available(),
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(str(e))
            return
        raise AssertionError("Baseline unexpectedly provides compression")
    rows_file = args.output.with_name("test-rows.json")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("test_bc250_hipsparselt.py")), "-v"],
        env=dict(os.environ, BC250_A04_RESULT_PATH=str(rows_file)),
    )
    rows = json.loads(rows_file.read_text()) if rows_file.exists() else []
    out = dict(
        status="passed" if result.returncode == 0 and len(rows) == 20 else "failed",
        returncode=result.returncode,
        results=rows,
        arch=torch.cuda.get_device_properties(0).gcnArchName,
        torch_version=torch.__version__,
        loaded_libraries=sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(n in line for n in ["libtorch_hip.so", "libhipsparselt.so"])
            }
        ),
    )
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    assert out["status"] == "passed"


if __name__ == "__main__":
    main()
