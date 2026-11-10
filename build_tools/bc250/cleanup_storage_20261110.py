#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retire inspected obsolete objects and independently archived proof copies only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

from external_kit_d53 import inventory

WORK = Path("/home/nik/bc250work")
TOOLS = WORK / "tools/storage-cleanup-20261110"
ROOT = WORK / "evidence/storage-cleanup-20261110"
WS = WORK / "rocm-10.2-gfx1013"
OBJECT_ROOTS = [
    WS / name / "build"
    for name in (
        "pytorch-distributed-b031",
        "pytorch-profiler-a09",
        "pytorch-profiler-order",
        "pytorch-rccl-b035",
        "pytorch-sparse-n213",
    )
]
OBJECT_ROOTS += [
    WS / name
    for name in (
        "tmp/ck-plugin-b0711",
        "tmp/miopen-plan02-cache-r1",
        "build/math-libs/rocPRIM_tests",
        "build/math-libs/rocThrust",
        "build/math-libs/hipCUB",
    )
]
PROOF_ROOTS = {
    "d47-r1": str(WORK / "evidence/source-d47-r1/replay"),
    "d47-r2": str(WORK / "evidence/source-d47-r2/replay"),
    "d50-r1": str(WORK / "evidence/source-d50-r1/replay"),
    "d52-r1": str(WORK / "evidence/source-d52-assembly-r1/assembled"),
    "d52-r2": str(WORK / "evidence/source-d52-assembly-r2/assembled"),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_record(path):
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    assert path.suffix == ".o" and path.parent.resolve() == path.parent
    assert any(path.is_relative_to(root) for root in OBJECT_ROOTS)
    with path.open("rb") as stream:
        assert stream.read(4) == b"\x7fELF"
    return dict(
        path=str(path),
        inode=info.st_ino,
        device=info.st_dev,
        bytes=info.st_size,
        allocated_bytes=info.st_blocks * 512,
        mode=info.st_mode,
        mtime_ns=info.st_mtime_ns,
        sha256=sha(path),
    )


def protected():
    paths = set()
    links = {}
    for base in (
        "/opt/bc250-stack/current",
        "/opt/bc250-stack/profiles/rccl-single",
        "/opt/bc250-stack/candidates/miopen-plan02-r4/profiles/daily",
    ):
        prefix = Path(base)
        links[base] = str(prefix.resolve())
        for name in (
            "runtime.json",
            "native-overrides.json",
            "rocm/lib/librocprofiler-sdk.so.1.4.2",
            "rocm/lib/libamdhip64.so",
            "rocm/lib/libMIOpen.so",
        ):
            p = prefix / name
            if p.is_file():
                paths.add(p)
        for p in (prefix / "torch/torch/lib").glob("*.so*"):
            if p.is_file():
                paths.add(p)
    for p in Path("/boot").glob("*"):
        if p.is_file():
            paths.add(p)
    for p in Path("/etc").glob("grub*"):
        if p.is_file():
            paths.add(p)
    for p in Path("/opt/bc250-stack/distributed").glob("*.json"):
        if p.is_file():
            paths.add(p)
    return dict(
        links=links,
        files={str(p): sha(p) for p in sorted(paths)},
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )


def assert_unused(objects, proofs):
    targets = set(objects)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "comm").read_text().strip()
            assert command not in ("ninja", "cc1", "cc1plus", "ld.lld"), (proc, command)
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                assert target not in targets and not any(
                    target == r or target.startswith(r + "/") for r in proofs
                ), (proc, target)
            mapped = (proc / "maps").read_text()
            assert not any(r + "/" in mapped for r in proofs), proc
        except FileNotFoundError:
            continue


def plan():
    ROOT.mkdir(exist_ok=False)
    started = time.monotonic()
    rows = []
    for root in OBJECT_ROOTS:
        assert root.is_dir() and root.resolve() == root
        for path in sorted(root.rglob("*.o")):
            if not path.is_symlink() and path.is_file() and path.stat().st_nlink == 1:
                rows.append(object_record(path))
    record = dict(
        status="planned",
        objects=rows,
        object_files=len(rows),
        object_allocated_bytes=sum(r["allocated_bytes"] for r in rows),
        free_before=shutil.disk_usage(WORK).free,
        protected=protected(),
        proof_roots=PROOF_ROOTS,
        seconds=time.monotonic() - started,
    )
    (ROOT / "plan.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ("objects", "protected")}
        )
    )


def apply():
    from board_preflight import preflight

    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    plan = json.loads((ROOT / "plan.json").read_text())
    receipt = json.loads((TOOLS / "archive-verified.json").read_text())
    manifest = json.loads((TOOLS / "archive-manifest.json").read_text())
    assert receipt["status"] == "passed" and receipt["manifest_sha256"] == sha(
        TOOLS / "archive-manifest.json"
    )
    assert plan["proof_roots"] == PROOF_ROOTS == manifest["roots"]
    result = dict(
        status="running", pid=os.getpid(), deleted=[], archive_receipt=receipt
    )
    (ROOT / "apply.json").write_text(json.dumps(result, indent=2) + "\n")
    with (WORK / "evidence/.board-reservation.lock").open("a") as board, Path(
        "/opt/bc250-stack/current/.runtime.lock"
    ).open("r") as lock:
        fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result["preflight"] = preflight("storage-cleanup-20261110")
            assert protected() == plan["protected"], "Protected runtime/boot drift"
            assert inventory(PROOF_ROOTS) == manifest, "Archived proof tree changed"
            for row in plan["objects"]:
                assert object_record(Path(row["path"])) == row
            assert_unused(
                [r["path"] for r in plan["objects"]], list(PROOF_ROOTS.values())
            )
            for row in plan["objects"]:
                path = Path(row["path"])
                info = path.lstat()
                assert (
                    info.st_ino == row["inode"]
                    and info.st_dev == row["device"]
                    and info.st_nlink == 1
                    and info.st_mtime_ns == row["mtime_ns"]
                )
                path.unlink()
                result["deleted"].append(dict(path=str(path), kind="obsolete-object"))
            for root in PROOF_ROOTS.values():
                path = Path(root)
                assert (
                    path.is_dir() and not path.is_symlink() and path.resolve() == path
                )
                shutil.rmtree(path)
                result["deleted"].append(dict(path=root, kind="locally-archived-proof"))
            result["protected_after"] = protected()
            assert result["protected_after"] == plan["protected"]
            result.update(
                status="passed",
                free_after=shutil.disk_usage(WORK).free,
                postflight=preflight("storage-cleanup-20261110-final"),
            )
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            (ROOT / "apply.json").write_text(json.dumps(result, indent=2) + "\n")
    for root in PROOF_ROOTS.values():
        (Path(root).parent / "RETIRED-PROOF-20261110.md").write_text(
            "# Retired redundant proof copy\n\nArchived and independently file-verified on the workstation before removal.\nRestore from kernel/artifacts/storage-cleanup-20261110/retired-source-proofs-r1.tar.gz; see its manifest roots for original locations.\nOriginal source kits, result JSON, logs and historical pass/fail records remain.\n"
        )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("deleted", "protected_after")}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["plan", "apply"])
    args = parser.parse_args()
    (plan if args.operation == "plan" else apply)()


if __name__ == "__main__":
    main()
