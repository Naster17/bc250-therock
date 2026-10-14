#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bounded native GPU checks of installed ROCm primitive headers."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    result = {"status": "failed", "cases": 0, "checks": []}
    start = time.monotonic()
    try:
        for binary, library, count in [
            ("rocprim", "rocPRIM", 42),
            ("hipcub", "hipCUB", 48),
            ("thrust", "rocThrust", 48),
        ]:
            run = subprocess.run(
                [str(args.binary_dir / binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            record = json.loads(run.stdout)
            if (
                record.get("status") != "passed"
                or record.get("architecture") != "gfx1013"
                or record.get("library") != library
                or record.get("cases") != count
                or record.get("streams") != 2
                or record.get("guards") is not True
            ):
                raise RuntimeError(f"Incomplete qualification: {record}")
            result["checks"].append(record)
            result["cases"] += count
        result["status"] = "passed"
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        result["seconds"] = round(time.monotonic() - start, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
