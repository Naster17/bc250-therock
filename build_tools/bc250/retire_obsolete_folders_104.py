#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retire three reviewed obsolete .104 folders after workstation archive verification."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

WORK = Path("/home/nik/bc250work")
NAMES = ["bc250-gfx1013-fix", "migration-105-20261023", "rocm-performance-20261013"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(8388608), b""):
            h.update(c)
    return h.hexdigest()


def inventory():
    result = {}
    for name in NAMES:
        root = WORK / name
        assert root.is_dir() and not root.is_symlink()
        for p in [root, *sorted(root.rglob("*"))]:
            key = str(p.relative_to(WORK))
            if p.is_symlink():
                result[key] = dict(kind="link", target=os.readlink(p))
            elif p.is_dir():
                result[key] = dict(kind="directory")
            else:
                assert p.is_file()
                result[key] = dict(kind="file", bytes=p.stat().st_size, sha256=sha(p))
    return result


def dependencies():
    findings = []
    roots = [
        "/etc",
        "/boot/grub",
        "/usr/local/bin",
        "/opt/bc250-stack/tools",
        "/opt/bc250-stack/distributed",
    ]
    files = [Path("/opt/bc250-stack/current/runtime.json")]
    for root in roots:
        files += list(Path(root).rglob("*"))
    for p in files:
        try:
            if p.is_symlink():
                content = os.readlink(p).encode()
            elif p.is_file() and p.stat().st_size < 2097152:
                content = p.read_bytes()
                if b"\x00" in content:
                    continue
            else:
                continue
            for name in NAMES:
                if ("bc250work/" + name).encode() in content:
                    findings.append(dict(path=str(p), candidate=name))
        except OSError:
            continue
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or proc.name == str(os.getpid()):
            continue
        try:
            content = (proc / "cmdline").read_bytes()
            for fd in (proc / "fd").iterdir():
                try:
                    content += os.readlink(fd).encode()
                except OSError:
                    pass
            for name in NAMES:
                if str(WORK / name).encode() in content:
                    findings.append(dict(pid=proc.name, candidate=name))
        except OSError:
            pass
    return findings


def protected():
    files = [
        Path("/opt/bc250-stack/current/runtime.json"),
        Path("/opt/bc250-stack/distributed/d01.json"),
        Path("/boot/grub/grub.cfg"),
        Path("/boot/grub/grubenv"),
    ]
    return {str(p): sha(p) for p in files if p.exists()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply-sha256")
    a = p.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    base = WORK / "evidence/folder-cleanup-20261104"
    base.mkdir(parents=True, exist_ok=True)
    archive = base / "obsolete-folders.tar.gz"
    plan = base / "plan.json"
    with Path("/opt/bc250-stack/current/.runtime.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        refs = dependencies()
        assert not refs, refs
        if not a.apply_sha256:
            assert not plan.exists() and not archive.exists()
            data = inventory()
            with tarfile.open(archive, "w:gz", compresslevel=1, dereference=False) as t:
                for name in NAMES:
                    t.add(WORK / name, arcname=name)
            result = dict(
                entries=data,
                archive_sha256=sha(archive),
                protected=protected(),
                references=refs,
            )
            plan.write_text(json.dumps(result, indent=2) + "\n")
            archive.chmod(0o644)
            print(
                json.dumps(
                    dict(
                        archive_sha256=result["archive_sha256"],
                        bytes=archive.stat().st_size,
                        entries=len(data),
                    )
                )
            )
            return
        data = json.loads(plan.read_text())
        assert a.apply_sha256 == data["archive_sha256"] == sha(archive)
        assert inventory() == data["entries"] and protected() == data["protected"]
        # Literal folders only: inventory, dependency audit, local-backup acknowledgement
        # and the common exclusive lock have all passed before any recursive removal.
        for name in NAMES:
            shutil.rmtree(WORK / name)
        assert protected() == data["protected"]
        result = dict(
            status="passed",
            removed=NAMES,
            logical_bytes=sum(x.get("bytes", 0) for x in data["entries"].values()),
            protected=data["protected"],
            references=refs,
            backup_sha256=data["archive_sha256"],
        )
        (base / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        archive.unlink()
        print(json.dumps(result))


if __name__ == "__main__":
    main()
