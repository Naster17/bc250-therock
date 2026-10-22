#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Broaden an independent RCCL candidate while preserving B03.3 and the live prefix."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

assert os.geteuid() == 0
preflight("rccl-b034-build")
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
assert shutil.disk_usage(WS).free > 14 * 1024**3
baseline = json.loads((WS / "logs/rccl-b033-qualification-r1/final.json").read_text())
for path, expected in baseline["protected_sha256"].items():
    assert sha(Path(path)) == expected, path
staged = WS / "staging/rccl-b034"
source = json.loads((staged / "source.json").read_text())
assert sha(staged / "rccl-source.tar") == source["source_archive_sha256"]
candidate = WS / "rccl-b034"
candidate.mkdir()
logs = WS / "logs/rccl-b034-build-r1"
logs.mkdir()
container = "bc250-rccl-b034-build-r1"
image = subprocess.check_output(
    [
        "podman",
        "image",
        "inspect",
        "localhost/bc250-pytorch-build:2.13-gfx1013",
        "--format",
        "{{.Id}}",
    ],
    text=True,
).strip()
report = dict(
    status="copying",
    pid=os.getpid(),
    container=container,
    image=image,
    source=source,
    protected_sha256=baseline["protected_sha256"],
    active_prefix_changed=False,
    old_candidate_preserved=True,
)
atomic_json(logs / "progress.json", report)
writer = subprocess.Popen(
    [
        "tar",
        "-C",
        str(WS / "rccl-b033"),
        "--exclude=./source",
        "--exclude=./install-r1",
        "--exclude=./readiness-r1",
        "-cf",
        "-",
        ".",
    ],
    stdout=subprocess.PIPE,
)
subprocess.run(
    ["tar", "-C", str(candidate), "-xf", "-"], stdin=writer.stdout, check=True
)
writer.stdout.close()
assert writer.wait() == 0
with tarfile.open(staged / "rccl-source.tar") as archive:
    archive.extractall(candidate / "source", filter="data")
command = [
    "podman",
    "run",
    "--name",
    container,
    "--network=none",
    "--cpus=4",
    "--memory=10g",
    "--memory-swap=12g",
]
for src, dst, mode in [
    (WS / "source", "/workspace/source", "ro"),
    (WS / "build", "/workspace/build", "ro"),
    (WS / "helpers", "/workspace/helpers", "ro"),
    (WS / "cache", "/workspace/cache", "rw"),
    (candidate, "/workspace/rccl", "rw"),
]:
    command += ["-v", f"{src}:{dst}:{mode}"]
command += [image, "bash", "/workspace/helpers/build_rccl_b034.sh"]
report.update(status="building", command=command)
atomic_json(logs / "progress.json", report)
try:
    with (logs / "build.log").open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, timeout=3600
        )
    report["returncode"] = result.returncode
    assert result.returncode == 0, result.returncode
    report["status"] = "built-not-qualified"
except BaseException as error:
    report.update(status="failed", error=repr(error))
    subprocess.run(
        ["podman", "stop", "--time", "10", container],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raise
finally:
    report["changed_hashes"] = [
        p for p, h in baseline["protected_sha256"].items() if sha(Path(p)) != h
    ]
    report["disk_available_bytes"] = shutil.disk_usage(WS).free
    atomic_json(logs / "final.json", report)
assert not report["changed_hashes"]
