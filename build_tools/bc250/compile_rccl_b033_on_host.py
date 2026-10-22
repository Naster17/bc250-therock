#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile the owned RCCL reference client without exposing GPU devices."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--attempt", default="r1")
parser.add_argument("--build-attempt", default="r3")
args = parser.parse_args()
assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
assert re.fullmatch(r"r[1-9][0-9]*", args.build_attempt)
assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
build = json.loads(
    (WS / ("logs/rccl-b033-build-" + args.build_attempt + "/final.json")).read_text()
)
assert build["status"] == "built-not-qualified" and not build["changed_hashes"]
candidate = WS / "rccl-b033"
logs = WS / ("logs/rccl-b033-client-build-" + args.attempt)
logs.mkdir()
command = [
    "podman",
    "run",
    "--name",
    "bc250-rccl-b033-client-" + args.attempt,
    "--network=none",
    "--cpus=2",
    "--memory=4g",
    "--memory-swap=5g",
]
for src, dst, mode in [
    (WS / "source", "/workspace/source", "ro"),
    (WS / "build", "/workspace/build", "ro"),
    (WS / "helpers", "/workspace/helpers", "ro"),
    (WS / "cache", "/workspace/cache", "rw"),
    (candidate, "/workspace/rccl", "rw"),
]:
    command += ["-v", f"{src}:{dst}:{mode}"]
command += [build["image"], "bash", "/workspace/helpers/compile_rccl_readiness_b033.sh"]
for path, expected in build["protected_sha256"].items():
    assert sha(Path(path)) == expected, path
with (logs / "build.log").open("w") as stream:
    result = subprocess.run(
        command, stdout=stream, stderr=subprocess.STDOUT, timeout=180
    )
report = dict(
    status="passed" if result.returncode == 0 else "failed",
    returncode=result.returncode,
    command=command,
)
atomic_json(logs / "final.json", report)
assert result.returncode == 0, result.returncode
baseline = json.loads(
    (WS / "logs/distributed-b032-qualification-r1/final.json").read_text()
)
for path, expected in baseline["protected_sha256"].items():
    assert sha(Path(path)) == expected, path
for path in (
    candidate / "readiness-r1",
    candidate / "install-r1/lib/librccl.so",
    candidate / "build-r1/CMakeCache.txt",
    WS / "staging/rccl-b033/source.json",
    WS / "staging/rccl-b033/rccl-source.tar",
):
    baseline["protected_sha256"][str(path)] = sha(path)
baseline["status"] = "uninstalled-candidate-preservation-baseline"
baseline["native_rccl_candidate"] = dict(
    library=str(candidate / "install-r1/lib/librccl.so"),
    client=str(candidate / "readiness-r1"),
    source=build["source"],
)
target = WS / "logs/rccl-b033-baseline-r1"
target.mkdir()
atomic_json(target / "final.json", baseline)
