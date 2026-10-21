#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Durably export only owned sparse artifacts/evidence; no user app inspection."""

import hashlib
import json
import os
from pathlib import Path
import tarfile

from qualify_training_n33_on_host import WS

qualified = json.loads(
    (WS / "logs/sparse-b021-qualification-r5/final.json").read_text()
)
assert qualified["status"] == "passed"
path = Path("/home/nik/bc250work/sparse-b021-evidence-r1.tar")
assert not path.exists()
with tarfile.open(path, "w") as archive:
    for p in sorted((WS / "logs").glob("sparse-b021-*")):
        archive.add(p, arcname="logs/" + p.name)
    for p in sorted(Path("/run").glob("sparse-b021-*/state.json")):
        archive.add(p, arcname="guards/" + p.parent.name + ".json")
    for p in (WS / "recovery/sparse-b021-before").iterdir():
        archive.add(p, arcname="recovery/" + p.name)
    for p in (WS / "helpers").iterdir():
        if p.is_file() and (
            "b021" in p.name
            or p.name
            in (
                "qualify_training_n33_on_host.py",
                "qualify_training_n39_on_host.py",
                "validate_torch_a04_on_host.py",
                "update_dev_prefix.py",
                "validate_sparse_n213.py",
                "test_bc250_sparse_triangular.py",
                "validate_priority2_math.py",
                "validate_cleanup_runtime.py",
            )
        ):
            archive.add(p, arcname="helpers/" + p.name)
    for name in (
        "bc250_sparse.py",
        "bc250_sparse.r1.py",
        "bc250_sparse.r2.py",
        "bc250_sparse.r3.py",
        "audit_sparse_b021.r1.py",
    ):
        archive.add(WS / "sparse-b021" / name, arcname="module/" + name)
    for context in sorted((WS / "sparse-b021").glob("runtime-context-*")):
        archive.add(context, arcname="contexts/" + context.name)
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
