#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Observe only the owned worker's children while its GIL is held in C++ JIT."""
import argparse
import json
from pathlib import Path
import os
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parent", type=int, required=True)
parser.add_argument("--stop", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
seen = set()
start = time.monotonic()
while (
    not args.stop.exists()
    and Path(f"/proc/{args.parent}").exists()
    and time.monotonic() - start < 500
):
    for task in Path(f"/proc/{args.parent}/task").glob("*"):
        try:
            children = (task / "children").read_text().split()
        except (OSError, ProcessLookupError):
            continue
        for pid in children:
            if int(pid) == os.getpid():
                continue
            try:
                exe = os.readlink(f"/proc/{pid}/exe")
                if Path(exe).name == "migraphx-hiprtc-driver":
                    seen.add(exe)
            except (OSError, ProcessLookupError):
                pass
    time.sleep(0.01)
args.output.write_text(
    json.dumps(dict(driver_executables=sorted(seen)), indent=2) + "\n"
)
