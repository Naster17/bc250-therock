#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D60: stage the qualified D58 release on .105 with inode-shared rocm.

Runs as root on bc500 (.105). Expects transfer chunks in TRANSFER (created by
the .104-side export). Builds an independent release tree: rocm hardlinked
from the retained immutable base with the two repaired files detached as
independent copies from the r4 candidate; profile .runtime.lock files
hardlink the migration-control inode. Never touches current or the base.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

VERSION = "10.2-gfx1013-20261111-138a66f52-host105"
STACK = Path("/opt/bc250-stack")
STAGING = STACK / "releases" / VERSION
BASE_ROCM = STACK / "releases/10.2-gfx1013-20261023-host105/rocm"
R4_ROCM = STACK / "candidates/miopen-plan02-r4/rocm"
TRANSFER = Path("/home/nik/bc250work/evidence/release-d60/transfer")
CONTROL_LOCK = STACK / "migration-control/.runtime.lock"
DETACHED = ("lib/libMIOpen.so.1.0", "lib/librocprofiler-sdk.so.1.4.2")
EXPECTED_RUNTIME = {
    "daily": "86d94f43f629b84572c2fc8639ef64beb96b32647439533f97260c7cd4e243fc",
    "rccl-single": "f183bd3f22ec55dcd1b1e10b44afb94f9edcb076b3224c34ad630f993fb0d67c",
}
MANIFEST = Path("/home/nik/bc250work/evidence/release-d60/rocm-release.sha256")


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hardlink_tree(src, dst):
    files = links = 0
    for dirpath, dirnames, filenames in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        target = dst / rel
        target.mkdir(parents=True, exist_ok=True)
        shutil.copystat(dirpath, target)
        for name in dirnames:
            p = Path(dirpath) / name
            if p.is_symlink():
                (target / name).symlink_to(os.readlink(p))
                links += 1
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                (target / name).symlink_to(os.readlink(p))
                links += 1
            else:
                os.link(p, target / name)
                files += 1
    return files, links


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    assert not STAGING.exists(), STAGING
    assert (STACK / "current").resolve() == (
        STACK / "candidates/miopen-plan02-r4/profiles/daily"
    )
    chunks = sorted(TRANSFER.glob("*.tar"))
    assert chunks, "no transfer chunks"
    STAGING.mkdir(parents=True)
    for chunk in chunks:
        subprocess.run(["tar", "-xf", str(chunk), "-C", str(STAGING)], check=True)

    files, links = hardlink_tree(BASE_ROCM, STAGING / "rocm")
    for rel in DETACHED:
        target = STAGING / "rocm" / rel
        target.unlink()
        shutil.copy2(R4_ROCM / rel, target)
        assert target.stat().st_ino != (R4_ROCM / rel).stat().st_ino
        assert target.stat().st_ino != (BASE_ROCM / rel).stat().st_ino

    control_ino = CONTROL_LOCK.stat().st_ino
    for profile in ("daily", "rccl-single"):
        lock = STAGING / "profiles" / profile / ".runtime.lock"
        if lock.exists() or lock.is_symlink():
            lock.unlink()
        os.link(CONTROL_LOCK, lock)
        assert lock.stat().st_ino == control_ino
        runtime = STAGING / "profiles" / profile / "runtime.json"
        assert sha(runtime) == EXPECTED_RUNTIME[profile], profile
        data = json.loads(runtime.read_text())
        assert data["validated"] and not data["mutable"]

    expected = {}
    for line in MANIFEST.read_text().splitlines():
        h, p = line.split(maxsplit=1)
        expected[p] = h
    bad = []
    for rel, want in expected.items():
        if sha(STAGING / "rocm" / rel) != want:
            bad.append(rel)
    assert not bad, bad[:10]
    for rel in DETACHED:
        assert (STAGING / "rocm" / rel).stat().st_nlink == 1

    violations = []
    for dirpath, dirnames, filenames in os.walk(STAGING):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                resolved = p.resolve()
                ok = resolved.is_relative_to(
                    STAGING.resolve()
                ) or resolved.is_relative_to(BASE_ROCM.resolve())
                if not ok:
                    violations.append(f"{p.relative_to(STAGING)} -> {os.readlink(p)}")
    assert not violations, violations

    receipt = {
        "task": "D60 staging",
        "version": VERSION,
        "staging": str(STAGING),
        "chunks": [c.name for c in chunks],
        "rocm_hardlinked_files": files,
        "rocm_symlinks": links,
        "rocm_detached": list(DETACHED),
        "rocm_manifest_verified": len(expected),
        "runtime_sha256": EXPECTED_RUNTIME,
        "lock_inode": control_ino,
        "external_symlink_violations": violations,
        "current_untouched": str((STACK / "current").resolve()),
        "status": "staged",
    }
    out = Path("/home/nik/bc250work/evidence/release-d60/staging.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    for chunk in chunks:
        chunk.unlink()
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
