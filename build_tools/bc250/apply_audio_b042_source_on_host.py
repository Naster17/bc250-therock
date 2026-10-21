#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Apply only the committed audio delta; preserve generated HIP files/builds."""

import argparse
import os
import subprocess
from qualify_training_n33_on_host import WS

assert os.geteuid() == 0
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--attempt", choices=("r2", "r3", "r4"), required=True)
args = parser.parse_args()
root = WS / "audio-b042"
source = root / "source"
old = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
).strip()
subprocess.run(
    [
        "git",
        "-C",
        str(source),
        "fetch",
        str(root / ("source-" + args.attempt + ".bundle")),
        "HEAD",
    ],
    check=True,
)
new = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "FETCH_HEAD"], text=True
).strip()
assert (
    subprocess.check_output(["git", "-C", str(source), "diff", old, new], text=True)
    == (root / ("source-" + args.attempt + ".patch")).read_text()
)
subprocess.run(
    [
        "git",
        "-C",
        str(source),
        "apply",
        "--check",
        str(root / ("source-" + args.attempt + ".patch")),
    ],
    check=True,
)
subprocess.run(
    [
        "git",
        "-C",
        str(source),
        "apply",
        str(root / ("source-" + args.attempt + ".patch")),
    ],
    check=True,
)
subprocess.run(["git", "-C", str(source), "update-ref", "HEAD", new, old], check=True)
print("AUDIO_SOURCE_UPDATE", old, new)
