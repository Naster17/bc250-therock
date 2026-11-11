#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D61: promote the .104 selection to the qualified release.

Runs as root on bc250 (.104) under the board EX reservation + current-prefix
EX runtime lock (inode checked). Switches current + the rccl-single pointer +
the distributed registry; records a recovery set for --rollback. The release
is immutable, so ordinary launches take no runtime lock; run smokes after
this helper exits (never launch mutable cohorts under the EX hold).
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from board_preflight import preflight

STACK = Path("/opt/bc250-stack")
WORK = Path("/home/nik/bc250work")
RELEASE = STACK / "releases/10.2-gfx1013-20261111-138a66f52"
CURRENT = STACK / "current"
RCCL_POINTER = STACK / "profiles/rccl-single"
REGISTRY = STACK / "distributed/d01.json"
RECOVERY = WORK / "evidence/release-d61/recovery"
OLD_DAILY = STACK / "candidates/sdk-d49-r1/profiles/daily"
OLD_RCCL = STACK / "candidates/sdk-d49-r1/profiles/rccl-single"
EXPECTED_RUNTIME = {
    "daily": "86d94f43f629b84572c2fc8639ef64beb96b32647439533f97260c7cd4e243fc",
    "rccl-single": "f183bd3f22ec55dcd1b1e10b44afb94f9edcb076b3224c34ad630f993fb0d67c",
}


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write(path, data, mode=0o644):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        stream.write(data)
    pending.chmod(mode)
    pending.replace(path)


def atomic_link(path, target):
    pending = path.with_name(path.name + ".d61-new")
    assert not pending.exists() and not pending.is_symlink()
    pending.symlink_to(target)
    pending.replace(path)


def locked():
    board = (WORK / "evidence/.board-reservation.lock").open("a")
    runtime = (CURRENT / ".runtime.lock").open("r")
    fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(runtime, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (
        os.fstat(runtime.fileno()).st_ino == (OLD_DAILY / ".runtime.lock").stat().st_ino
    )
    return board, runtime


def promote():
    assert not RECOVERY.exists(), "recovery record already exists"
    board, runtime = locked()
    try:
        info = preflight("d61-promote")
        assert CURRENT.resolve() == OLD_DAILY
        assert RCCL_POINTER.resolve() == OLD_RCCL
        for profile, digest in EXPECTED_RUNTIME.items():
            assert sha(RELEASE / "profiles" / profile / "runtime.json") == digest
        RECOVERY.mkdir(parents=True)
        before = {
            "current": os.readlink(CURRENT),
            "rccl_pointer": os.readlink(RCCL_POINTER),
            "registry_sha256": sha(REGISTRY),
            "preflight": info,
        }
        atomic_write(RECOVERY / "registry.before.json", REGISTRY.read_bytes())
        atomic_write(
            RECOVERY / "before.json", (json.dumps(before, indent=2) + "\n").encode()
        )
        atomic_link(CURRENT, RELEASE / "profiles/daily")
        atomic_link(RCCL_POINTER, RELEASE / "profiles/rccl-single")
        record = json.loads(REGISTRY.read_text())
        record["release"] = str(RELEASE / "profiles/rccl-single")
        record["runtime_sha256"] = EXPECTED_RUNTIME["rccl-single"]
        atomic_write(REGISTRY, (json.dumps(record, indent=2) + "\n").encode())
        atomic_write(RECOVERY / "registry.after.json", REGISTRY.read_bytes())
        assert CURRENT.resolve() == RELEASE / "profiles/daily"
        assert RCCL_POINTER.resolve() == RELEASE / "profiles/rccl-single"
        after = {
            "current": str(CURRENT.resolve()),
            "rccl_pointer": str(RCCL_POINTER.resolve()),
            "registry_sha256": sha(REGISTRY),
            "status": "promoted",
        }
        atomic_write(
            RECOVERY / "after.json", (json.dumps(after, indent=2) + "\n").encode()
        )
        print(json.dumps(after, indent=2))
    finally:
        runtime.close()
        board.close()


def rollback():
    board, runtime = locked()
    try:
        preflight("d61-rollback")
        before = json.loads((RECOVERY / "before.json").read_text())
        assert sha(RECOVERY / "registry.before.json") == before["registry_sha256"]
        atomic_link(CURRENT, before["current"])
        atomic_link(RCCL_POINTER, before["rccl_pointer"])
        atomic_write(REGISTRY, (RECOVERY / "registry.before.json").read_bytes())
        assert CURRENT.resolve() == OLD_DAILY
        assert RCCL_POINTER.resolve() == OLD_RCCL
        state = {"current": str(CURRENT.resolve()), "status": "rolled-back"}
        atomic_write(
            RECOVERY / "rollback.json", (json.dumps(state, indent=2) + "\n").encode()
        )
        print(json.dumps(state, indent=2))
    finally:
        runtime.close()
        board.close()


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.rollback:
        rollback()
    else:
        promote()


if __name__ == "__main__":
    main()
