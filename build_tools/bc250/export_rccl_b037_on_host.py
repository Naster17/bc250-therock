#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export owned distributed build, native payload, qualification and recovery evidence."""

import hashlib
import json
import os
from pathlib import Path
import tarfile

from qualify_training_n33_on_host import WS

qualified = json.loads((WS / "logs/rccl-b037-qualification-r2/final.json").read_text())
assert qualified["status"] == "passed"
assert sum(g["checks"] for g in qualified["gates"]) == 1441
path = Path("/home/nik/bc250work/rccl-b037-evidence-r1.tar")
assert not path.exists()
source = WS / "pytorch-rccl-b037"
with tarfile.open(path, "w") as archive:
    for p in sorted((WS / "logs").glob("rccl-b037-*")):
        archive.add(p, arcname="logs/" + p.name)
    for p in sorted(Path("/run").glob("rccl-b037-*/state.json")):
        archive.add(p, arcname="guards/" + p.parent.name + ".json")
    for directory in ("rccl-b037-before", "rccl-b037-reduce-before"):
        archive.add(WS / "recovery" / directory, arcname="recovery/" + directory)
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b037" in p.name
            or p.name
            in (
                "qualify_training_n33_on_host.py",
                "qualify_training_n39_on_host.py",
                "qualify_rccl_b036_on_host.py",
                "qualify_distributed_b031_on_host.py",
                "validate_torch_a04_on_host.py",
                "update_dev_prefix.py",
                "torch_toolchain.cmake",
                *qualified["source_sha256"],
            )
        ):
            archive.add(p, arcname="helpers/" + p.name)
    for p in (WS / "staging/rccl-b037").iterdir():
        if p.suffix in (".json", ".patch"):
            archive.add(p, arcname="staging/" + p.name)
    for p in sorted((source / "torch/lib").glob("*.so")):
        archive.add(p.resolve(), arcname="native/torch/lib/" + p.name)
    for p in sorted((source / "torch").glob("_C*.so")):
        archive.add(p.resolve(), arcname="native/torch/" + p.name)
    for name in (
        "GIT_REVISION",
        "torch/version.py",
        "build/CMakeCache.txt",
        "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp",
        "torch/csrc/cuda/nccl.cpp",
        "test/distributed/test_nccl.py",
        "test/distributed/test_c10d_ops_nccl.py",
    ):
        archive.add(source / name, arcname="build/" + name)
    archive.add(
        Path("/opt/bc250-stack/current/runtime.json"), arcname="installed/runtime.json"
    )
    archive.add(
        WS / "rccl-b037/candidate-release-r2/runtime.json",
        arcname="candidate/runtime.json",
    )
    for p in sorted((WS / "logs").glob("training-n39-resource-rccl-b037*.json")):
        archive.add(p, arcname="resources/" + p.name)
with path.open("rb") as stream:
    os.fsync(stream.fileno())
fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
os.fsync(fd)
os.close(fd)
with path.open("rb") as stream:
    print(
        path.stat().st_size,
        hashlib.file_digest(stream, "sha256").hexdigest(),
        flush=True,
    )
