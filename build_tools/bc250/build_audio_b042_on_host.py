#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Isolated native torchaudio build under the exclusive development lock."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--attempt", default="r1")
args = parser.parse_args()
assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
root = WS / "audio-b042"
logs = WS / ("logs/audio-b042-build-" + args.attempt)
logs.mkdir()
prior = json.loads((WS / "logs/vision-b041-qualification-r3/final.json").read_text())
protected = dict(prior["protected_sha256"])
admitted = {
    "/home/nik/finetune-n7/AGENTS.md": "9ca3401720469dbfbc4a9e97d112bc88feef9e8c7dfb1e174c3d5df24c6e9e9d"
}
changes = {}
for name, expected in protected.items():
    actual = sha(Path(name))
    if actual != expected:
        assert admitted.get(name) == actual, name
        changes[name] = dict(previous=expected, entry=actual)
        protected[name] = actual
assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 6 * 1024**3
if not (root / "source").exists():
    with tarfile.open(root / "source.tar") as archive:
        archive.extractall(root, filter="data")
revision = subprocess.check_output(
    ["git", "-C", str(root / "source"), "rev-parse", "HEAD"], text=True
).strip()
assert (
    json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())["image"]
    == "a0b85fa88c9ecfbec88a8774f19db0fbc18fcfeaa086a4e402b69aa99ba55e27"
)
report = dict(
    status="building",
    source_revision=revision,
    source_archive_sha256=sha(root / "source.tar"),
    protected_sha256=protected,
    entry_document_drift=changes,
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    resources=dict(
        jobs=4, memory="10g", memory_swap="12g", ccache="20G", gpu_devices=False
    ),
    build_image="acdb6dc417950345b995d772a7c23285dfb1a54aac8fcb33b26d5cf46ce0e6cf",
)
if args.attempt == "r1":
    entry = WS / "logs/audio-b042-entry-r1"
    entry.mkdir()
    prior.update(
        status="preservation-only-documentation-entry-drift",
        protected_sha256=protected,
        entry_document_drift=changes,
        note="User Burn/Vulkan documentation additions preserved; not renewed application qualification.",
    )
    atomic_json(entry / "final.json", prior)
command = [
    "podman",
    "run",
    "--rm",
    "--name",
    "audio-b042-build-" + args.attempt,
    "--network=none",
    "--memory=10g",
    "--memory-swap=12g",
    "--cpus=8",
    "--entrypoint",
    "bash",
]
for source, target, mode in (
    (root, "/workspace/audio-b042", "rw"),
    (WS / "helpers", "/workspace/helpers", "ro"),
    (WS / "source", "/workspace/source", "ro"),
    (WS / "cache", "/workspace/cache", "rw"),
    (Path("/opt/bc250-stack/current/torch"), "/workspace/pytorch-work", "ro"),
    (Path("/opt/bc250-stack/current/rocm"), "/workspace/build/dist/rocm", "ro"),
):
    command += ["-v", f"{source}:{target}:{mode}"]
command += ["-e", "AUDIO_WHEEL_DIR=/workspace/audio-b042/wheels-" + args.attempt]
command += [report["build_image"], "/workspace/helpers/build_audio_b042.sh"]
atomic_json(logs / "command.json", command)
atomic_json(logs / "progress.json", report)
try:
    with (logs / "wheel.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600
        )
    wheel = next((root / ("wheels-" + args.attempt)).glob("torchaudio-*.whl"))
    report.update(status="passed", wheel=str(wheel), wheel_sha256=sha(wheel))
except Exception as error:
    report.update(status="failed", error=repr(error))
    raise
finally:
    report["changed_protected_hashes"] = [
        name for name, h in protected.items() if sha(Path(name)) != h
    ]
    assert not report["changed_protected_hashes"]
    atomic_json(logs / "final.json", report)
