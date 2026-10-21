#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export owned evidence with an explicit file/directory durability barrier."""

import hashlib
import json
import os
from pathlib import Path
import tarfile
from qualify_training_n33_on_host import WS

assert (
    json.loads((WS / "logs/audio-b042-qualification-r6/final.json").read_text())[
        "status"
    ]
    == "passed"
)
path = Path("/home/nik/bc250work/audio-b042-evidence-r2.tar")
assert not path.exists()
with tarfile.open(path, "w") as archive:
    for p in sorted((WS / "logs").glob("audio-b042-*")):
        archive.add(p, arcname="logs/" + p.name)
    for p in (WS / "recovery/audio-b042-before").iterdir():
        archive.add(p, arcname="recovery/" + p.name)
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b042" in p.name
            or p.name
            in (
                "qualify_training_n33_on_host.py",
                "validate_torch_a04_on_host.py",
                "update_dev_prefix.py",
            )
        ):
            archive.add(p, arcname="helpers/" + p.name)
    for p in (WS / "audio-b042/runtime-candidate-r1").iterdir():
        archive.add(p, arcname="runtime-candidate/" + p.name)
    for name in (
        "verification.json",
        "libamdhip64-symbols.so",
        "original-link-commands.txt",
        "link-command.json",
    ):
        archive.add(WS / "audio-b042/hip-symbols" / name, arcname="hip-symbols/" + name)
    for name in ("rocvirtual.cpp", "hostcall_queue_b042"):
        archive.add(WS / "audio-b042" / name, arcname="native/" + name)
    archive.add(
        Path("/opt/bc250-stack/current/runtime.json"), arcname="installed/runtime.json"
    )
with path.open("rb") as f:
    os.fsync(f.fileno())
fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(fd)
finally:
    os.close(fd)
print(
    path.stat().st_size,
    hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    flush=True,
)
