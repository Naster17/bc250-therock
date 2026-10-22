#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export owned distributed build, native payload, qualification and recovery evidence."""

import hashlib
import json
import os
from pathlib import Path
import tarfile

from qualify_training_n33_on_host import WS

qualified = json.loads(
    (WS / "logs/distributed-b031-qualification-r3/final.json").read_text()
)
assert qualified["status"] == "passed"
path = Path("/home/nik/bc250work/distributed-b031-evidence-r1.tar")
assert not path.exists()
source = WS / "pytorch-distributed-b031"
with tarfile.open(path, "w") as archive:
    for p in sorted((WS / "logs").glob("distributed-b031-*")):
        archive.add(p, arcname="logs/" + p.name)
    for p in sorted(Path("/run").glob("distributed-b031-*/state.json")):
        archive.add(p, arcname="guards/" + p.parent.name + ".json")
    for p in (WS / "recovery/distributed-b031-before").iterdir():
        if p.is_file() and not p.is_symlink():
            archive.add(p, arcname="recovery/" + p.name)
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b031" in p.name
            or p.name
            in (
                "qualify_training_n33_on_host.py",
                "qualify_training_n39_on_host.py",
                "validate_torch_a04_on_host.py",
                "update_dev_prefix.py",
                "torch_toolchain.cmake",
                *qualified["source_sha256"],
            )
        ):
            archive.add(p, arcname="helpers/" + p.name)
    for p in (WS / "staging/distributed-b031").iterdir():
        if p.suffix in (".json", ".patch"):
            archive.add(p, arcname="staging/" + p.name)
    for p in sorted((source / "torch/lib").glob("*.so")):
        archive.add(p.resolve(), arcname="native/torch/lib/" + p.name)
    for p in sorted((source / "torch").glob("_C*.so")):
        archive.add(p.resolve(), arcname="native/torch/" + p.name)
    for name in ("GIT_REVISION", "torch/version.py", "build/CMakeCache.txt"):
        archive.add(source / name, arcname="build/" + name)
    archive.add(
        Path("/opt/bc250-stack/current/runtime.json"), arcname="installed/runtime.json"
    )
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
