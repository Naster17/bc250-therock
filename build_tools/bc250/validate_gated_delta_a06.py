#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded gfx1013 gated-delta preparation and full recurrence references."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--installed", action="store_true")
    args = ap.parse_args()
    rows_file = args.output.with_name("rows.json")
    env = dict(os.environ, BC250_A06_ROWS=str(rows_file))
    if args.installed:
        env["BC250_A06_INSTALLED"] = "1"
    code = subprocess.call(
        [
            sys.executable,
            str(Path(__file__).with_name("test_bc250_gated_delta.py")),
            "-v",
        ],
        env=env,
    )
    rows = json.loads(rows_file.read_text()) if rows_file.exists() else []
    result = dict(
        status="passed" if code == 0 and len(rows) == 40 else "failed",
        returncode=code,
        results=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
