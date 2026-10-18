#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded native profiler traces, references and lifecycle tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = args.output.parent / "traces"
    env = dict(os.environ, BC250_A09_OUTPUT=str(traces))
    code = subprocess.call(
        [
            sys.executable,
            str(Path(__file__).with_name("test_bc250_profiler_a09.py")),
            "-v",
        ],
        env=env,
    )
    rows_path = traces / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    result = dict(
        status="passed" if code == 0 and len(rows) == 7 else "failed",
        returncode=code,
        results=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
