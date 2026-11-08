#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded candidate regression, leaving the installed provider unchanged."""
import hashlib
import json
from pathlib import Path
import sys

sys.path[:0] = [
    "/home/nik/bc250work/tools/plan01",
    "/home/nik/bc250work/rocm-10.2-gfx1013/helpers",
]
from qualify_rccl_b038_on_host import gate

ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
meta = json.loads((ws / "tmp/ck-plugin-b0711/native-fix-r2/final.json").read_text())
library = Path(meta["library"])
assert hashlib.sha256(library.read_bytes()).hexdigest() == meta["library_sha256"]
runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
results = []
try:
    for profile in ["daily", "rccl-single"]:
        cmd = [
            "python3",
            "/home/nik/bc250work/tools/plan02/launch_miopen_plan02.py",
            "--runtime-profile",
            profile,
            "--shm-size",
            "256m",
            "/home/nik/bc250work/tools/plan02/validate_miopen_plan02_backward.py",
            "--library",
            str(library),
            "--sha",
            meta["library_sha256"],
        ]
        results.append(gate("plan02-backward-" + profile + "-r1", cmd, 12, runtime))
finally:
    Path("/home/nik/bc250work/evidence/plan02/backward-final.json").write_text(
        json.dumps(dict(gates=results), indent=2) + "\n"
    )
