#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Durably export only owned codec artifacts/evidence; no user app inspection."""

import hashlib
import json
import os
from pathlib import Path
import tarfile

from qualify_training_n33_on_host import WS

qualified = json.loads((WS / "logs/video-b044-qualification-r3/final.json").read_text())
assert qualified["status"] == "passed"
path = Path("/home/nik/bc250work/video-b044-evidence-r1.tar")
assert not path.exists()
with tarfile.open(path, "w") as archive:
    for p in sorted((WS / "logs").glob("video-b044-*")):
        archive.add(p, arcname="logs/" + p.name)
    for p in sorted(Path("/run").glob("video-b044-*/state.json")):
        archive.add(p, arcname="guards/" + p.parent.name + ".json")
    for p in (WS / "recovery/video-b044-before").iterdir():
        archive.add(p, arcname="recovery/" + p.name)
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b044" in p.name
            or p.name
            in (
                "qualify_training_n33_on_host.py",
                "qualify_training_n39_on_host.py",
                "validate_torch_a04_on_host.py",
                "update_dev_prefix.py",
                "validate_audio_b042.py",
            )
        ):
            archive.add(p, arcname="helpers/" + p.name)
    for name in (
        "build-packages.txt",
        "build-python-packages.json",
        "source-provenance.json",
    ):
        archive.add(WS / "video-b044" / name, arcname="build/" + name)
    archive.add(WS / "video-b044/wheels-r1", arcname="wheels")
    archive.add(
        Path("/opt/bc250-stack/current/runtime.json"), arcname="installed/runtime.json"
    )
with path.open("rb") as stream:
    os.fsync(stream.fileno())
fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(fd)
finally:
    os.close(fd)
with path.open("rb") as stream:
    print(
        path.stat().st_size,
        hashlib.file_digest(stream, "sha256").hexdigest(),
        flush=True,
    )
