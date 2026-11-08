#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve old KDB inputs, require corrected cold/warm output and cache-key separation."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path[:0] = [
    "/home/nik/bc250work/tools/plan01",
    "/home/nik/bc250work/rocm-10.2-gfx1013/helpers",
]
from qualify_rccl_b038_on_host import gate


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and not p.name.endswith(("-wal", "-shm"))
    }


def rows(root):
    result = {}
    for db in root.rglob("*.ukdb"):
        with closing(sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True)) as c:
            for name, args, sha in c.execute(
                "select kernel_name,kernel_args,kernel_hash from kern_db"
            ):
                result[(str(db.relative_to(root)), name, args)] = sha
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument(
        "--old-cache-root",
        type=Path,
        default=Path("/home/nik/bc250work/evidence/plan02"),
    )
    a = p.parse_args()
    stage = json.loads((a.root / "stage.json").read_text())
    assert stage["cache_key_version"] == 1
    work = Path("/home/nik/bc250work")
    out = work / "evidence" / ("plan02-cache-migration-" + a.attempt)
    out.mkdir()
    os.environ["PATH"] = str(a.root / "test-tools") + ":" + os.environ["PATH"]
    runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    report = dict(
        status="running",
        gates=[],
        checks=[],
        library_sha256=stage["library_sha256"],
        cache_key_version=1,
        profiles={},
    )
    try:
        for profile in ["daily", "rccl-single"]:
            original = a.old_cache_root / ("plan02-old-" + profile + "-r1") / "cache"
            source = hashes(original)
            oldrows = rows(original)
            assert oldrows
            folder = out / profile
            shutil.copytree(original, folder / "cache")
            for path in [folder, *folder.rglob("*")]:
                os.chown(path, 1000, 1000)
            for phase in ["cold", "warm"]:
                before_rows = rows(folder / "cache")
                cmd = [
                    str(a.root / "tools/bc250_torch.py"),
                    "--runtime-profile",
                    profile,
                    "--env",
                    "MIOPEN_LOG_LEVEL=6",
                    "--env",
                    "MIOPEN_CUSTOM_CACHE_DIR=" + str(folder / "cache"),
                    "--env",
                    "MIOPEN_USER_DB_PATH=" + str(folder / "db"),
                    "/home/nik/bc250work/tools/plan02/validate_miopen_plan02.py",
                    "--library-sha",
                    stage["library_sha256"],
                ]
                report["gates"].append(
                    gate(
                        "plan02-cache-" + a.attempt + "-" + profile + "-" + phase,
                        cmd,
                        1,
                        runtime,
                    )
                )
                current = rows(folder / "cache")
                assert all(current.get(k) == v for k, v in oldrows.items())
                tagged = [
                    k for k in current if "BC250_DIRECT_FWD_FP32_ACCUM_CACHE=1" in k[2]
                ]
                assert len(tagged) >= 6, tagged
                assert hashes(original) == source
                if phase == "warm":
                    assert before_rows == current, "warm cache records changed"
                    label = "plan02-cache-" + a.attempt + "-" + profile + "-" + phase
                    log = (
                        Path("/home/nik/bc250work/rocm-10.2-gfx1013/logs")
                        / label
                        / "gpu.log"
                    ).read_text()
                    assert "Saving binary for:" not in log
                    hits = [
                        l
                        for l in log.splitlines()
                        if 'Successfully loaded binary for: "MIOpenConvDirUniHip.cpp.o"'
                        in l
                    ]
                    assert len(hits) >= 12
                    assert (
                        sum("BC250_DIRECT_FWD_FP32_ACCUM_CACHE=1" in l for l in hits)
                        >= 6
                    )
                report["checks"].append(
                    profile + "-" + phase + "-precision-correct-old-entries-retained"
                )
            report["profiles"][profile] = dict(
                original_cache=str(original),
                original_hashes=source,
                old_rows=len(oldrows),
                final_rows=len(current),
                versioned_rows=len(tagged),
                warm_logical_records_unchanged=True,
                warm_binary_saves=0,
            )
        report["status"] = "passed"
    except BaseException as e:
        report.update(status="failed", error=repr(e))
        raise
    finally:
        (out / "final.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
