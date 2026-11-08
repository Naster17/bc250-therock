#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove restored pointer/registry bytes and mapped original native arithmetic."""
import argparse
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--original-sha", required=True)
    a = p.parse_args()
    work = Path("/home/nik/bc250work")
    out = work / "evidence" / ("plan02-rollback-" + a.attempt)
    out.mkdir()
    os.environ["PATH"] = str(a.root / "test-tools") + ":" + os.environ["PATH"]
    report = dict(status="running", gates=[], checks=[])
    try:
        before = json.loads(a.before.read_text())
        for name, target in before["pointers"].items():
            path = Path(name)
            if target is None:
                assert not path.exists() and not path.is_symlink()
            else:
                assert path.is_symlink() and os.readlink(path) == target
        assert (
            Path("/opt/bc250-stack/distributed/d01.json").read_bytes()
            == Path(before["distributed_backup"]).read_bytes()
        )
        lib = Path("/opt/bc250-stack/current/rocm/lib/libMIOpen.so.1.0")
        with lib.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == a.original_sha
        report["checks"].append("original-pointers-D01-bytes-native-library-restored")
        runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
        profiles = (
            ["daily", "rccl-single"] if os.uname().nodename == "bc250" else ["daily"]
        )
        for profile in profiles:
            scratch = out / profile
            scratch.mkdir()
            os.chown(scratch, 1000, 1000)
            cmd = ["/usr/local/bin/bc250-torch"]
            if os.uname().nodename == "bc250":
                cmd += ["--runtime-profile", profile]
            cmd += [
                "--env",
                "MIOPEN_USER_DB_PATH=" + str(scratch / "db"),
                "--env",
                "MIOPEN_CUSTOM_CACHE_DIR=" + str(scratch / "cache"),
                str(work / "tools/plan02/validate_miopen_plan02.py"),
                "--library-sha",
                a.original_sha,
                "--expect-old",
            ]
            report["gates"].append(
                gate("plan02-rollback-" + a.attempt + "-" + profile, cmd, 1, runtime)
            )
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (out / "final.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
