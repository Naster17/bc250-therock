#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage a pure Python explicit sparse solve adapter image, preserving active runtime."""

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
parser.add_argument("--attempt", choices=("r1", "r2", "r3", "r4"), default="r1")
args = parser.parse_args()
assert os.geteuid() == 0
preflight("sparse-b021-stage")
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
prior = json.loads((WS / "logs/fft-b011-qualification-r3/final.json").read_text())
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
root = WS / "sparse-b021"
context = root / ("runtime-context-" + args.attempt)
context.mkdir()
shutil.copy2(root / "bc250_sparse.py", context / "bc250_sparse.py")
logs = WS / ("logs/sparse-b021-stage-" + args.attempt)
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
            ("localhost/bc250-stack:sparse-b021-" + args.attempt),
            "--file",
            str(WS / "helpers/Containerfile.sparse-runtime-b021"),
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
        ("localhost/bc250-stack:sparse-b021-" + args.attempt),
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
manifest["sparse_b021"] = dict(
    version="0.1.0",
    module_sha256=sha(root / "bc250_sparse.py"),
    scope="Explicit bounded GPU dense-LU CSR fallback candidate; qualification pending; no global dispatch change.",
)
atomic_json(candidate / "runtime.json", manifest)
backup = WS / "recovery/sparse-b021-before/runtime.json"
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
        module_sha256=sha(root / "bc250_sparse.py"),
    ),
)
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
