#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage a pure Python explicit FFT adapter image, preserving active runtime."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--attempt", choices=("r1", "r2", "r3"), default="r1")
args = parser.parse_args()
assert os.geteuid() == 0
preflight("fft-b011-stage")
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
prior = json.loads((WS / "logs/image-b045-qualification-r3/final.json").read_text())
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
root = WS / "fft-b011"
context = root / ("runtime-context-" + args.attempt)
context.mkdir()
shutil.copy2(root / "bc250_fft.py", context / "bc250_fft.py")
logs = WS / ("logs/fft-b011-stage-" + args.attempt)
logs.mkdir()
(root / "tmp").mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(root / "tmp")
with (logs / "image.log").open("w") as stream:
    subprocess.run(
        [
            "podman",
            "build",
            "--pull=never",
            "--network=none",
            "--memory=1g",
            "--tag",
            ("localhost/bc250-stack:fft-b011-" + args.attempt),
            "--file",
            str(WS / "helpers/Containerfile.fft-runtime-b011"),
            str(context),
        ],
        check=True,
        stdout=stream,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
image = subprocess.check_output(
    [
        "podman",
        "image",
        "inspect",
        ("localhost/bc250-stack:fft-b011-" + args.attempt),
        "--format",
        "{{.Id}}",
    ],
    text=True,
).strip()
active = Path("/opt/bc250-stack/current").resolve(strict=True)
candidate = root / ("candidate-release-" + args.attempt)
candidate.mkdir()
for entry in active.iterdir():
    if entry.name != "runtime.json":
        (candidate / entry.name).symlink_to(entry)
manifest = json.loads((active / "runtime.json").read_text())
previous = manifest["image"]
manifest["image"] = image
manifest["fft_b011"] = dict(
    version="0.1.0",
    module_sha256=sha(root / "bc250_fft.py"),
    scope="Explicit BF16-to-FP32 FFT helper candidate; qualification pending; no global dispatch change.",
)
atomic_json(candidate / "runtime.json", manifest)
backup = WS / "recovery/fft-b011-before/runtime.json"
if not backup.exists():
    shutil.copy2(active / "runtime.json", backup)
atomic_json(
    logs / "final.json",
    dict(
        status="passed",
        image=image,
        previous_image=previous,
        candidate_release=str(candidate),
        candidate_runtime_sha256=sha(candidate / "runtime.json"),
        module_sha256=sha(root / "bc250_fft.py"),
    ),
)
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
