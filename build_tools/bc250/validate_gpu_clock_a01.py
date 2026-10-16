#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only KFD/shader clock correlation probe."""

import argparse
import json
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--reference", action="store_true")
args = ap.parse_args()
r = subprocess.run(
    [str(Path(__file__).with_name("miopen_a01_clock_probe"))]
    + (["--reference"] if args.reference else []),
    text=True,
    capture_output=True,
    check=True,
    timeout=30,
)
rows = [json.loads(line) for line in r.stdout.splitlines()]
args.output.write_text(json.dumps(dict(samples=rows), indent=2) + "\n")
print(r.stdout, end="")
