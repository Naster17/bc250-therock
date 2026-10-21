#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an independent debugger image; original candidate stays untouched."""

import json
import os
import subprocess
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS

assert os.geteuid() == 0
root = WS / "audio-b042"
os.environ["TMPDIR"] = str(root / "tmp-r1")
with (WS / "logs/audio-b042-debug-image-r1.log").open("w") as log:
    subprocess.run(
        [
            "podman",
            "build",
            "--network=host",
            "--pull=never",
            "--tag",
            "localhost/bc250-audio-debug:b042-r1",
            "--file",
            str(WS / "helpers/Containerfile.audio-debug-b042"),
            str(root / "runtime-context-r1"),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=600,
    )
image = subprocess.check_output(
    [
        "podman",
        "image",
        "inspect",
        "localhost/bc250-audio-debug:b042-r1",
        "--format",
        "{{.Id}}",
    ],
    text=True,
).strip()
source = root / "candidate-release-r1"
dest = root / "debug-release-r1"
dest.mkdir()
for p in source.iterdir():
    if p.name != "runtime.json":
        (dest / p.name).symlink_to(p)
manifest = json.loads((source / "runtime.json").read_text())
manifest["image"] = image
atomic_json(dest / "runtime.json", manifest)
