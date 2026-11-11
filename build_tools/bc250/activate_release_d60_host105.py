#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D60: atomically activate (or roll back) the staged release on .105.

Runs as root on bc500 (.105) through the migration-control lock:
  update_dev_prefix.py --release /opt/bc250-stack/migration-control -- \
      python3 activate_release_d60_host105.py [--rollback]
Switches current + the rccl-single profile pointer + the distributed registry;
preserves the old release, image, caches, boot health and launchers.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

STACK = Path("/opt/bc250-stack")
VERSION = "10.2-gfx1013-20261111-138a66f52-host105"
STAGING = STACK / "releases" / VERSION
CURRENT = STACK / "current"
RCCL_POINTER = STACK / "profiles/rccl-single"
REGISTRY = STACK / "distributed/d01.json"
RECOVERY = Path("/home/nik/bc250work/evidence/release-d60/recovery")
EXPECTED_RUNTIME = {
    "daily": "86d94f43f629b84572c2fc8639ef64beb96b32647439533f97260c7cd4e243fc",
    "rccl-single": "f183bd3f22ec55dcd1b1e10b44afb94f9edcb076b3224c34ad630f993fb0d67c",
}
OLD_DAILY = STACK / "candidates/miopen-plan02-r4/profiles/daily"
OLD_RCCL = STACK / "candidates/miopen-plan02-r4/profiles/rccl-single"


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
    pending = path.with_name(path.name + ".d60-new")
    assert not pending.exists() and not pending.is_symlink()
    pending.symlink_to(target)
    pending.replace(path)


def activate():
    assert not RECOVERY.exists(), "recovery record already exists"
    assert CURRENT.resolve() == OLD_DAILY
    assert RCCL_POINTER.resolve() == OLD_RCCL
    for profile, digest in EXPECTED_RUNTIME.items():
        assert sha(STAGING / "profiles" / profile / "runtime.json") == digest
    RECOVERY.mkdir(parents=True)
    before = {
        "current": os.readlink(CURRENT),
        "rccl_pointer": os.readlink(RCCL_POINTER),
        "registry_sha256": sha(REGISTRY),
    }
    atomic_write(RECOVERY / "registry.before.json", REGISTRY.read_bytes())
    atomic_write(
        RECOVERY / "before.json", (json.dumps(before, indent=2) + "\n").encode()
    )
    atomic_link(CURRENT, STAGING / "profiles/daily")
    atomic_link(RCCL_POINTER, STAGING / "profiles/rccl-single")
    record = json.loads(REGISTRY.read_text())
    record["release"] = str(STAGING / "profiles/rccl-single")
    record["runtime_sha256"] = EXPECTED_RUNTIME["rccl-single"]
    atomic_write(REGISTRY, (json.dumps(record, indent=2) + "\n").encode())
    assert CURRENT.resolve() == STAGING / "profiles/daily"
    assert RCCL_POINTER.resolve() == STAGING / "profiles/rccl-single"
    after = {
        "current": str(CURRENT.resolve()),
        "rccl_pointer": str(RCCL_POINTER.resolve()),
        "registry_sha256": sha(REGISTRY),
        "status": "activated",
    }
    atomic_write(RECOVERY / "after.json", (json.dumps(after, indent=2) + "\n").encode())
    print(json.dumps(after, indent=2))


def rollback():
    before = json.loads((RECOVERY / "before.json").read_text())
    assert sha(RECOVERY / "registry.before.json") == before["registry_sha256"]
    atomic_link(CURRENT, before["current"])
    atomic_link(RCCL_POINTER, before["rccl_pointer"])
    atomic_write(REGISTRY, (RECOVERY / "registry.before.json").read_bytes())
    assert CURRENT.resolve() == OLD_DAILY
    assert RCCL_POINTER.resolve() == OLD_RCCL
    after = {"current": str(CURRENT.resolve()), "status": "rolled-back"}
    atomic_write(
        RECOVERY / "rollback.json", (json.dumps(after, indent=2) + "\n").encode()
    )
    print(json.dumps(after, indent=2))


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.rollback:
        rollback()
    else:
        activate()


if __name__ == "__main__":
    main()
