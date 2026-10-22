#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export the isolated native RCCL payload, failed builds and bounded GPU evidence."""

import hashlib
import json
import os
from pathlib import Path
import tarfile

from qualify_training_n33_on_host import WS

qualified = json.loads((WS / "logs/rccl-b034-qualification-r1/final.json").read_text())
assert qualified["status"] == "passed"
assert (
    len(qualified["gates"]) == 4 and sum(g["checks"] for g in qualified["gates"]) == 629
)
path = Path("/home/nik/bc250work/rccl-b034-evidence-r1.tar")
assert not path.exists()
candidate = WS / "rccl-b034"
with tarfile.open(path, "w") as archive:
    for p in sorted((WS / "logs").glob("rccl-b034-*")):
        archive.add(p, arcname="logs/" + p.name)
    for name in ("rccl-b034-build", "rccl-b034-qualification"):
        p = WS / "logs" / ("training-n39-resource-" + name + ".json")
        archive.add(p, arcname="resources/" + p.name)
    for p in sorted(Path("/run").glob("rccl-b034-*/state.json")):
        archive.add(p, arcname="guards/" + p.parent.name + ".json")
    for directory in ("rccl-b034-before",):
        archive.add(WS / "recovery" / directory, arcname="recovery/" + directory)
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b034" in p.name
            or p.name
            in (
                "qualify_training_n39_on_host.py",
                "update_dev_prefix.py",
                *qualified["source_sha256"],
            )
        ):
            archive.add(p, arcname="helpers/" + p.name)
    for p in (WS / "staging/rccl-b034").iterdir():
        if p.name != "rccl-source.tar":
            archive.add(p, arcname="staging/" + p.name)
    for src, dest in (
        (candidate / "install-r1/lib/librccl.so", "native/librccl.so"),
        (candidate / "readiness-r1", "native/readiness-r1"),
        (candidate / "install-r1/include/rccl/rccl.h", "native/include/rccl/rccl.h"),
        (candidate / "build-r1/CMakeCache.txt", "build/CMakeCache.txt"),
        (Path("/opt/bc250-stack/current/runtime.json"), "installed/runtime.json"),
    ):
        archive.add(src.resolve(), arcname=dest)
with path.open("rb") as stream:
    os.fsync(stream.fileno())
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
os.fsync(fd)
os.close(fd)
print(path.stat().st_size, digest, flush=True)
