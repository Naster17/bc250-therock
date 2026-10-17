#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run A06 precision correction or A07 head256 device TestCase gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--component", choices=["a06", "a07"], required=True)
    ap.add_argument("--candidate", type=Path)
    args = ap.parse_args()
    a06 = args.component == "a06"
    prefix = "BC250_A06" if a06 else "BC250_A07"
    script = "test_bc250_gated_delta.py" if a06 else "test_bc250_attention_a07.py"
    rows_path = args.output.with_name("rows.json")
    env = dict(os.environ)
    env[prefix + "_ROWS"] = str(rows_path)
    if args.candidate:
        env[prefix + "_CANDIDATE"] = str(args.candidate)
    elif a06:
        env["BC250_A06_INSTALLED"] = "1"
    code = subprocess.call(
        [sys.executable, str(Path(__file__).with_name(script)), "-v"], env=env
    )
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    expected = 42 if a06 else 36
    result = dict(
        status="passed" if code == 0 and len(rows) == expected else "failed",
        returncode=code,
        results=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
