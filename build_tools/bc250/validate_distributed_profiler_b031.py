#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retain the CPU-first native profiler gate across the distributed Torch build."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
raw = args.output.with_name("cpu-first-raw.json")
subprocess.run(
    [
        sys.executable,
        str(Path(__file__).with_name("validate_profiler_order.py")),
        "--order",
        "cpu-first",
        "--output",
        str(raw),
    ],
    check=True,
    timeout=120,
)
r = json.loads(raw.read_text())
assert r["status"] == "passed" and len(r["captures"]) == 5 and len(r["references"]) == 1
r["checks"] = ["five-native-captures-and-cpu-output-gradient-copy-references"]
args.output.write_text(json.dumps(r, indent=2) + "\n")
