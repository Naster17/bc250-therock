#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reversible cohort pointer transaction under stable EX runtime/reservation locks."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, "/home/nik/bc250work/tools/plan01")
from board_preflight import preflight

BASE = Path("/opt/bc250-stack")
D01 = BASE / "distributed/d01.json"


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_bytes(p, data):
    tmp = p.with_name(p.name + ".plan02-new")
    assert not tmp.exists()
    tmp.write_bytes(data)
    tmp.chmod(0o644)
    os.replace(tmp, p)


def link(p, target):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".plan02-new")
    assert not tmp.exists()
    tmp.symlink_to(target)
    os.replace(tmp, p)


def pointers(root):
    result = {
        str(BASE / "native-cohort"): str(root),
        str(BASE / "current"): str(root / "profiles/daily"),
        str(BASE / "profiles/rccl-single"): str(root / "profiles/rccl-single"),
    }
    if (root / "tools/bc250_ck.py").exists():
        for name, module in [
            ("bc250-ck", "bc250_ck.py"),
            ("bc250-migraphx", "bc250_migraphx.py"),
            ("bc250-ort", "bc250_ort.py"),
        ]:
            result["/usr/local/bin/" + name] = str(root / "tools" / module)
    return result


def restore(record):
    for name, target in record["pointers"].items():
        path = Path(name)
        if target is None:
            assert path.is_symlink()
            path.unlink()
        else:
            link(path, target)
    atomic_bytes(D01, Path(record["distributed_backup"]).read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--action", choices=["activate", "rollback"], required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--cache-qualification", type=Path)
    p.add_argument("--from-transaction", type=Path)
    a = p.parse_args()
    assert re.fullmatch(r"[a-z0-9-]+", a.label)
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    root = a.root.resolve()
    assert root.is_relative_to(BASE) and not root.is_symlink()
    stage = json.loads((root / "stage.json").read_text())
    for profile, digest in stage["profiles"].items():
        assert sha(root / "profiles" / profile / "runtime.json") == digest
    assert (
        sha(root / "profiles/daily/rocm/lib/libMIOpen.so.1.0")
        == stage["library_sha256"]
    )
    if a.action == "activate":
        assert (
            stage.get("cache_key_version") == 1
        ), "Cache-unsafe cohorts must not be activated"
        q = json.loads(a.qualification.read_text())
        assert q["status"] == "passed"
        assert len(q["gates"]) >= (10 if os.uname().nodename == "bc250" else 4)
        assert q["stage"]["library_sha256"] == stage["library_sha256"]
        assert a.cache_qualification, "Retained-cache qualification required"
        cache = json.loads(a.cache_qualification.read_text())
        assert cache["status"] == "passed" and cache["cache_key_version"] == 1
        assert cache["library_sha256"] == stage["library_sha256"]
        assert len(cache["gates"]) >= 4
        assert set(cache["profiles"]) == {"daily", "rccl-single"}
        for result in cache["profiles"].values():
            assert result["warm_logical_records_unchanged"]
            assert result["warm_binary_saves"] == 0
            assert result["versioned_rows"] >= 6
        for group in ["tools", "registries"]:
            for name, digest in stage.get(group, {}).items():
                assert sha(Path(name)) == digest
    preflight("plan02-" + a.action)
    report_dir = root / "transactions" / a.label
    report_dir.mkdir(parents=True)
    journal = dict(
        status="preparing",
        action=a.action,
        root=str(root),
        pid=os.getpid(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    record = json.loads(D01.read_text())
    # The old and new current profiles must use the SAME stable lock inode.
    oldlock = (
        BASE / "current/.runtime.lock"
        if os.uname().nodename == "bc250"
        else BASE / "migration-control/.runtime.lock"
    )
    newlock = root / "profiles/daily/.runtime.lock"
    assert (oldlock.stat().st_dev, oldlock.stat().st_ino) == (
        newlock.stat().st_dev,
        newlock.stat().st_ino,
    )
    paths = [
        Path("/home/nik/bc250work/evidence/.board-reservation.lock"),
        oldlock,
        Path(record["lock"]),
    ]
    try:
        with ExitStack() as stack:
            seen = set()
            for path in paths:
                fd = stack.enter_context(path.open("a" if path == paths[0] else "r"))
                key = (os.fstat(fd.fileno()).st_dev, os.fstat(fd.fileno()).st_ino)
                if key in seen:
                    continue
                seen.add(key)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            preflight("plan02-" + a.action + "-locked")
            before = {}
            for name in pointers(root):
                path = Path(name)
                assert not path.exists() or path.is_symlink(), name
                before[name] = os.readlink(path) if path.is_symlink() else None
            backup = report_dir / "distributed.before.json"
            backup.write_bytes(D01.read_bytes())
            journal.update(
                pointers=before, distributed_backup=str(backup), status="locked"
            )
            atomic_bytes(
                report_dir / "before.json",
                (json.dumps(journal, indent=2) + "\n").encode(),
            )
            try:
                if a.action == "rollback":
                    old = json.loads(a.from_transaction.read_text())
                    restore(old)
                else:
                    value = dict(
                        record,
                        release=str(root / "profiles/rccl-single"),
                        runtime_sha256=stage["profiles"]["rccl-single"],
                    )
                    value["native_precision_cohort"] = dict(
                        root=str(root),
                        MIOpen_sha256=stage["library_sha256"],
                        scope="PLAN-02 native forward accumulation; Torch/RCCL unchanged",
                    )
                    atomic_bytes(D01, (json.dumps(value, indent=2) + "\n").encode())
                    for name, target in pointers(root).items():
                        link(Path(name), target)
                journal["status"] = "passed"
            except BaseException:
                restore(journal)
                journal["status"] = "restored-after-failure"
                raise
    except BaseException as exc:
        journal.update(error=repr(exc))
        raise
    finally:
        atomic_bytes(
            report_dir / "result.json", (json.dumps(journal, indent=2) + "\n").encode()
        )
    print(report_dir / "result.json", flush=True)


if __name__ == "__main__":
    main()
