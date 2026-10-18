#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""One fresh-process profiler-order case per guarded container."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--order",
        choices=("cpu-first", "gpu-first", "cpu-cold", "masked"),
        required=True,
    )
    parser.add_argument("--expect-missing", action="store_true")
    args = parser.parse_args()
    traces = args.output.parent / "traces"
    command = [
        sys.executable,
        str(Path(__file__).with_name("test_bc250_profiler_order.py")),
        "--worker",
        args.order,
        "--output",
        str(traces),
    ]
    if args.expect_missing:
        command.append("--expect-missing")
    env = dict(os.environ)
    if args.order == "masked":
        env.update(HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1")
    code = subprocess.call(command, env=env)
    result = (
        json.loads((traces / "result.json").read_text())
        if (traces / "result.json").exists()
        else dict(status="failed")
    )
    result["returncode"] = code
    if code != 0:
        result["status"] = "failed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
