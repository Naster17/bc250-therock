#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create an isolated audio runtime candidate; never modify the active prefix."""

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
parser.add_argument("--attempt", choices=("r1", "r2"), default="r2")
args = parser.parse_args()
assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
preflight("audio-b042-stage-" + args.attempt)
root = WS / "audio-b042"
build_attempt = "r3" if args.attempt == "r1" else "r4"
prior = json.loads(
    (WS / ("logs/audio-b042-build-" + build_attempt + "/final.json")).read_text()
)
assert prior["status"] == "passed"
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
context = root / ("runtime-context-" + args.attempt)
(context / "wheels").mkdir(parents=True)
audio = next(
    (root / ("wheels" if args.attempt == "r1" else "wheels-r4")).glob(
        "torchaudio-*.whl"
    )
)
assert sha(audio) == prior["wheel_sha256"]
shutil.copy2(audio, context / "wheels" / audio.name)
(root / ("tmp-" + args.attempt)).mkdir()
os.environ["TMPDIR"] = str(root / ("tmp-" + args.attempt))
logs = WS / ("logs/audio-b042-stage-" + args.attempt)
logs.mkdir()
with (logs / "image.log").open("w") as log:
    subprocess.run(
        [
            "podman",
            "build",
            "--pull=never",
            "--network=none",
            "--tag",
            ("localhost/bc250-stack:audio-b042-" + args.attempt),
            "--file",
            str(WS / "helpers/Containerfile.audio-b042"),
            str(context),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=180,
    )
image = subprocess.check_output(
    [
        "podman",
        "image",
        "inspect",
        ("localhost/bc250-stack:audio-b042-" + args.attempt),
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
manifest["image"] = image
manifest["torchaudio_b042"] = dict(
    version="2.11.0+gfx1013",
    source_revision=prior["source_revision"],
    wheel_sha256=sha(audio),
    scope="Candidate; qualification pending. Native HIP signal operators. File APIs require separately uninstalled TorchCodec.",
)
atomic_json(candidate / "runtime.json", manifest)
if args.attempt == "r1":
    backup = WS / "recovery/audio-b042-before"
    backup.mkdir()
    shutil.copy2(active / "runtime.json", backup / "runtime.json")
    shutil.copy2(
        WS / "helpers/qualify_training_n33_on_host.py",
        backup / "qualify_training_n33_on_host.py",
    )
    os.replace(
        WS / "helpers/qualify_training_n33.audio-next.py",
        WS / "helpers/qualify_training_n33_on_host.py",
    )
atomic_json(
    logs / "final.json",
    dict(
        status="passed",
        candidate_release=str(candidate),
        image=image,
        previous_image=prior.get(
            "previous_image", json.loads((active / "runtime.json").read_text())["image"]
        ),
        wheels={audio.name: sha(audio)},
        candidate_runtime_sha256=sha(candidate / "runtime.json"),
        protected_sha256=prior["protected_sha256"],
    ),
)
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
