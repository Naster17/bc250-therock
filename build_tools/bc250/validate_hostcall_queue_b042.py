#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Direct native hostcall initialization with guarded CPU-referenced outputs."""

import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
command = ["/home/nik/bc250work/rocm-10.2-gfx1013/audio-b042/hostcall_queue_b042"]
result = subprocess.run(command, capture_output=True, text=True, timeout=120)
report = dict(
    status="passed" if result.returncode == 0 else "failed",
    checks=(
        ["direct-cold-hostcall-60-streams-cpu-reference-guards"]
        if result.returncode == 0
        else []
    ),
    returncode=result.returncode,
    stdout=result.stdout,
    stderr=result.stderr,
)
args.output.write_text(json.dumps(report, indent=2) + "\n")
assert result.returncode == 0, report
assert (
    result.stdout.strip() == "HOSTCALL_QUEUE_PASS rounds=3 streams=60 guarded_words=384"
)
