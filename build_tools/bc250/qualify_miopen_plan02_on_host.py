#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare installed and isolated precision repair; no native prefix writes."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path[:0] = [
    "/home/nik/bc250work/tools/plan01",
    "/home/nik/bc250work/rocm-10.2-gfx1013/helpers",
]
from qualify_rccl_b038_on_host import gate


def main():
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    tools = Path("/home/nik/bc250work/tools/plan02")
    out = Path("/home/nik/bc250work/evidence/plan02")
    out.mkdir(exist_ok=True)
    runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    old = Path("/opt/bc250-stack/current/rocm/lib/libMIOpen.so.1.0")
    new = Path(
        json.loads((ws / "tmp/ck-plugin-b0711/native-fix-r2/final.json").read_text())[
            "library"
        ]
    )
    protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [old, new]}
    report = dict(status="running", protected=protected, gates=[])
    try:
        for version, library in [("new", new)]:
            for profile in ["daily", "rccl-single"]:
                label = "plan02-" + version + "-" + profile + "-r3"
                scratch = out / label
                scratch.mkdir()
                os.chown(scratch, 1000, 1000)
                command = [
                    "python3",
                    str(tools / "launch_miopen_plan02.py"),
                    "--runtime-profile",
                    profile,
                    "--shm-size",
                    "256m",
                ]
                env = dict(
                    MIOPEN_USER_DB_PATH=str(scratch / "db"),
                    MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "cache"),
                    OMP_NUM_THREADS="4",
                )
                for k, v in env.items():
                    command += ["--env", k + "=" + v]
                command += [
                    str(tools / "validate_miopen_plan02.py"),
                    "--library-sha",
                    protected[str(library)],
                ]
                if version == "new":
                    command += ["--candidate-library", str(library)]
                if version == "old":
                    command += ["--expect-old"]
                report["gates"].append(gate(label, command, 1, runtime))
                (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        assert all(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() == s
            for p, s in protected.items()
        )
        report["status"] = "passed"
    except BaseException as e:
        report.update(status="failed", error=repr(e))
        raise
    finally:
        (out / "candidate-final.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
