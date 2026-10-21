#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Durably export only owned codec artifacts/evidence; no user app inspection."""

import hashlib
import json
import os
from pathlib import Path
import tarfile
import subprocess

from qualify_training_n33_on_host import WS

qualified = json.loads((WS / "logs/image-b045-qualification-r3/final.json").read_text())
assert qualified["status"] == "passed"
path = Path("/home/nik/bc250work/image-b045-evidence-r1.tar")
assert not path.exists()
payload = WS / "image-b045/runtime-sysdeps.tar"
assert not payload.exists()
image = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())["image"]
with payload.open("wb") as stream:
    subprocess.run(
        [
            "podman",
            "run",
            "--rm",
            "--read-only",
            "--network=none",
            "--memory=256m",
            "--cpus=1",
            "--entrypoint",
            "tar",
            image,
            "-cf",
            "-",
            "-C",
            "/",
            "opt/bc250-codec-sysdeps",
            "etc/ld.so.conf.d/bc250-codec-images.conf",
        ],
        check=True,
        stdout=stream,
        timeout=60,
    )
with tarfile.open(path, "w") as archive:
    archive.add(payload, arcname="runtime-sysdeps.tar")
    for p in sorted((WS / "logs").glob("image-b045-*")):
        archive.add(p, arcname="logs/" + p.name)
    for p in sorted(Path("/run").glob("image-b045-*/state.json")):
        archive.add(p, arcname="guards/" + p.parent.name + ".json")
    for p in (WS / "recovery/image-b045-before").iterdir():
        archive.add(p, arcname="recovery/" + p.name)
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b045" in p.name
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
        archive.add(WS / "image-b045" / name, arcname="build/" + name)
    archive.add(WS / "image-b045/wheels-r1", arcname="wheels")
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
