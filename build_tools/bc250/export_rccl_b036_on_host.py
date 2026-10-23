#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export B03.6 evidence; native payloads remain in verified B03.5/B03.4 artifacts."""

import hashlib
import json
import os
from pathlib import Path
import tarfile

from qualify_training_n33_on_host import WS

qualified = json.loads((WS / "logs/rccl-b036-qualification-r1/final.json").read_text())
assert qualified["status"] == "passed"
assert (
    len(qualified["gates"]) == 21
    and sum(g["checks"] for g in qualified["gates"]) == 1365
)
path = Path("/home/nik/bc250work/rccl-b036-evidence-r1.tar")
assert not path.exists()
with tarfile.open(path, "w") as archive:
    for p in sorted((WS / "logs").glob("rccl-b036-*")):
        archive.add(p, arcname="logs/" + p.name)
    for p in sorted((WS / "logs").glob("training-n39-resource-rccl-b036*.json")):
        archive.add(p, arcname="resources/" + p.name)
    for p in sorted(Path("/run").glob("rccl-b036-*/state.json")):
        archive.add(p, arcname="guards/" + p.parent.name + ".json")
    archive.add(WS / "recovery/rccl-b036-before", arcname="recovery/rccl-b036-before")
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b036" in p.name
            or p.name
            in (
                *qualified["source_sha256"],
                "qualify_distributed_b031_on_host.py",
                "qualify_training_n39_on_host.py",
            )
        ):
            archive.add(p, arcname="helpers/" + p.name)
    for p, name in (
        (Path("/opt/bc250-stack/current/runtime.json"), "installed/runtime.json"),
        (WS / "rccl-b035/candidate-release-r3/runtime.json", "candidate/runtime.json"),
        (WS / "pytorch-rccl-b035/GIT_REVISION", "candidate/GIT_REVISION"),
        (WS / "pytorch-rccl-b035/build/CMakeCache.txt", "candidate/CMakeCache.txt"),
    ):
        archive.add(p, arcname=name)
with path.open("rb") as stream:
    os.fsync(stream.fileno())
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
os.fsync(fd)
os.close(fd)
print(path.stat().st_size, digest, flush=True)
