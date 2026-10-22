#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Expose a built, uninstalled distributed Torch candidate to guarded tests."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--cpu-only-staging",
    action="store_true",
    help="Prepare files while other GPU work runs; does not qualify or expose the GPU",
)
args = parser.parse_args()
assert os.geteuid() == 0
if args.cpu_only_staging:
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    assert available > 8 * 1024**3, available
else:
    preflight("distributed-b031-stage")
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
built = json.loads((WS / "logs/distributed-b031-build/final.json").read_text())
assert built["status"] == "built-not-qualified"
for path, expected in built["protected_sha256"].items():
    assert sha(Path(path)) == expected, path
source = Path(built["candidate"])
staged = WS / "staging/distributed-b031"
dependency_files = json.loads((staged / "dependency-files.json").read_text())
delta = json.loads((staged / "gloo-source-delta.json").read_text())
verified_dependencies = {}
for name, files in dependency_files.items():
    root = (
        WS
        / "pytorch-sparse-followup/third_party"
        / {
            "gloo": "gloo",
            "tensorpipe": "tensorpipe",
            "libuv": "tensorpipe/third_party/libuv",
        }[name]
    )
    for filename, expected in files.items():
        if name == "gloo" and filename == delta["changed_file"]:
            expected = delta["build_source_sha256"]
        assert sha(root / filename) == expected, (name, filename)
    verified_dependencies[name] = len(files)
for name, expected in built["native_sha256"].items():
    assert sha(source / "torch/lib" / name) == expected, name
active = Path("/opt/bc250-stack/current").resolve(strict=True)
candidate = WS / "distributed-b031/candidate-release-r1"
candidate.mkdir(parents=True)
for entry in active.iterdir():
    if entry.name not in ("runtime.json", "torch"):
        (candidate / entry.name).symlink_to(entry)
(candidate / "torch").symlink_to(source)
manifest = json.loads((active / "runtime.json").read_text())
manifest["torch_revision"] = built["source"]["torch_revision"]
manifest["distributed_b031"] = dict(
    status="candidate; qualification pending",
    source=built["source"],
    dependency_files_verified=verified_dependencies,
    existing_gloo_source_delta=delta,
    build_evidence=str(WS / "logs/distributed-b031-build/final.json"),
    native_sha256=built["native_sha256"],
    scope="Local CPU Gloo process groups first; RCCL/MPI off; no distributed GPU or multi-board claim.",
)
atomic_json(candidate / "runtime.json", manifest)
backup = WS / "recovery/distributed-b031-before"
assert backup.is_dir()
(backup / "runtime.json").write_bytes((active / "runtime.json").read_bytes())
(backup / "torch-link.txt").write_text(os.readlink(active / "torch") + "\n")
(backup / "torch").symlink_to((active / "torch").resolve())
logs = WS / "logs/distributed-b031-stage-r1"
logs.mkdir()
atomic_json(
    logs / "final.json",
    dict(
        status="passed",
        cpu_only_staging=args.cpu_only_staging,
        gpu_qualification_pending=True,
        candidate_release=str(candidate),
        candidate_runtime_sha256=sha(candidate / "runtime.json"),
        candidate_torch=str(source),
        native_sha256=built["native_sha256"],
        old_torch_target=os.readlink(active / "torch"),
        old_runtime_sha256=sha(active / "runtime.json"),
        image=manifest["image"],
    ),
)
