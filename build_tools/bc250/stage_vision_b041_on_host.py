#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create an isolated vision runtime candidate; never modify the active prefix."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
root = WS / "vision-b041"
prior = json.loads((WS / "logs/vision-b041-build-r3/final.json").read_text())
assert prior["status"] == "passed"
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
assert json.loads((root / "wheel-repair-r3.json").read_text())["status"] == "passed"
context = root / "runtime-context-r2"
(context / "wheels").mkdir(parents=True)
vision = next((root / "repaired-wheels-r3").glob("*.whl"))
pillow = root / prior["pillow"]["filename"]
assert sha(pillow) == prior["pillow"]["sha256"]
for wheel in (vision, pillow):
    shutil.copy2(wheel, context / "wheels" / wheel.name)
os.environ["TMPDIR"] = str(root / "tmp-r3")
logs = WS / "logs/vision-b041-stage-r2"
logs.mkdir()
with (logs / "image.log").open("w") as log:
    subprocess.run(
        [
            "podman",
            "build",
            "--pull=never",
            "--network=none",
            "--tag",
            "localhost/bc250-stack:vision-b041-r2",
            "--file",
            str(WS / "helpers/Containerfile.vision-runtime-b041"),
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
        "localhost/bc250-stack:vision-b041-r2",
        "--format",
        "{{.Id}}",
    ],
    text=True,
).strip()
active = Path("/opt/bc250-stack/current").resolve(strict=True)
candidate = root / "candidate-release-r2"
candidate.mkdir()
for entry in active.iterdir():
    if entry.name != "runtime.json":
        (candidate / entry.name).symlink_to(entry)
manifest = json.loads((active / "runtime.json").read_text())
manifest["image"] = image
manifest["torchvision_b041"] = dict(
    version="0.28.0+gfx1013",
    source_revision=prior["source_revision"],
    wheel_sha256=sha(vision),
    pillow=prior["pillow"],
    scope="Candidate; qualification pending. CPU codecs; native gfx1013 image operators.",
)
atomic_json(candidate / "runtime.json", manifest)
atomic_json(
    logs / "final.json",
    dict(
        status="passed",
        candidate_release=str(candidate),
        image=image,
        previous_image=prior.get(
            "previous_image", json.loads((active / "runtime.json").read_text())["image"]
        ),
        wheels={w.name: sha(w) for w in (vision, pillow)},
        candidate_runtime_sha256=sha(candidate / "runtime.json"),
        protected_sha256=prior["protected_sha256"],
    ),
)
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
