#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export owned single-board Gloo GPU qualification without changing the runtime."""

import hashlib
import json
import os
from pathlib import Path
import tarfile

from qualify_training_n33_on_host import WS

qualified = json.loads(
    (WS / "logs/distributed-b032-qualification-r1/final.json").read_text()
)
assert qualified["status"] == "passed"
path = Path("/home/nik/bc250work/distributed-b032-evidence-r1.tar")
assert not path.exists()
with tarfile.open(path, "w") as archive:
    for p in sorted((WS / "logs").glob("distributed-b032-*")):
        archive.add(p, arcname="logs/" + p.name)
    archive.add(
        WS / "logs/training-n39-resource-distributed-b032.json", arcname="resource.json"
    )
    for p in sorted(Path("/run").glob("distributed-b032-*/state.json")):
        archive.add(p, arcname="guards/" + p.parent.name + ".json")
    for p in (WS / "recovery/distributed-b032-before").iterdir():
        archive.add(p, arcname="recovery/" + p.name)
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b032" in p.name
            or p.name
            in (
                "qualify_training_n39_on_host.py",
                "update_dev_prefix.py",
                *qualified["source_sha256"],
            )
        ):
            archive.add(p, arcname="helpers/" + p.name)
    archive.add(
        Path("/opt/bc250-stack/current/runtime.json"), arcname="installed/runtime.json"
    )
with path.open("rb") as stream:
    os.fsync(stream.fileno())
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
os.fsync(fd)
os.close(fd)
print(path.stat().st_size, digest, flush=True)
