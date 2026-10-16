#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run reusable training API TestCase references in the guarded GPU launcher."""
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
    env = dict(os.environ, BC250_A05_ROWS=str(rows_file))
    if args.installed:
        env["BC250_A05_INSTALLED"] = "1"
    code = subprocess.call([sys.executable, str(Path(__file__).with_name("test_training_a05.py")), "-v"], env=env)
    rows = json.loads(rows_file.read_text()) if rows_file.exists() else []
    result = dict(status="passed" if code == 0 and len(rows) == 69 else "failed", returncode=code, results=rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
