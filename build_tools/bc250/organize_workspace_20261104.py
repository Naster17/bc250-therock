#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retire workstation-verified transfer copies; retain stable development paths."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert os.geteuid() == 0
    host = "104" if os.uname().nodename == "bc250" else "105"
    work = Path("/home/nik/bc250work")
    plan = json.loads((work / "cleanup-20261104-plan.json").read_text())
    lock = Path(
        "/opt/bc250-stack/current/.runtime.lock"
        if host == "104"
        else "/opt/bc250-stack/migration-control/.runtime.lock"
    )
    journal = work / "cleanup-20261104-result.json"
    assert not journal.exists()
    current = Path("/opt/bc250-stack/current")
    protected = [
        current / "runtime.json",
        Path("/opt/bc250-stack/distributed/d01.json"),
    ]
    if host == "105":
        protected += [
            Path("/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105/runtime.json")
        ]
    before = {str(p): sha(p) for p in protected}
    selected = str(current.resolve())
    result = dict(
        host=host,
        before=before,
        selected=selected,
        deleted=[],
        moved=[],
        status="running",
    )
    with lock.open("r") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        # Validate the whole allowlist before deleting the first file.
        for entry in plan[host]:
            p = Path(entry["remote"])
            assert p.is_file() and not p.is_symlink() and p.is_relative_to(work)
            assert sha(p) == entry["sha256"] and p.stat().st_size == entry["bytes"]
        for entry in plan[host]:
            Path(entry["remote"]).unlink()
            result["deleted"].append(entry)
            journal.write_text(json.dumps(result, indent=2) + "\n")
        # Existing absolute paths remain valid through relative compatibility links.
        groups = {"tools/distributed-d01": [], "evidence/distributed-d01": []}
        for p in sorted(work.iterdir()):
            if p.is_symlink():
                continue
            if p.is_file() and ("distributed" in p.name) and p.suffix == ".py":
                groups["tools/distributed-d01"].append(p)
            elif (
                p.is_file()
                and p.name.startswith("distributed-")
                and p.name.endswith("-controller.log")
            ):
                groups["evidence/distributed-d01"].append(p)
            elif p.is_dir() and p.name.startswith(
                (
                    "distributed-d01-",
                    "distributed-d02-",
                    "distributed-d03-",
                    "distributed-d04-",
                )
            ):
                if p.name not in [
                    "distributed-d01-import",
                    "distributed-d01-export",
                    "distributed-d01-tools",
                ]:
                    groups["evidence/distributed-d01"].append(p)
        for group, paths in groups.items():
            dest = work / group
            dest.mkdir(parents=True, exist_ok=True)
            for p in paths:
                target = dest / p.name
                assert not target.exists()
                p.rename(target)
                p.symlink_to(
                    os.path.relpath(target, p.parent),
                    target_is_directory=target.is_dir(),
                )
                result["moved"].append(dict(old=str(p), new=str(target)))
        after = {str(p): sha(p) for p in protected}
        assert before == after and str(current.resolve()) == selected
        result.update(
            status="passed",
            after=after,
            removed_logical_bytes=sum(x["bytes"] for x in result["deleted"]),
        )
        journal.write_text(json.dumps(result, indent=2) + "\n")
        journal.chmod(0o644)
    print(
        json.dumps(
            dict(
                host=host,
                status=result["status"],
                files=len(result["deleted"]),
                moved=len(result["moved"]),
                bytes=result["removed_logical_bytes"],
            )
        )
    )


if __name__ == "__main__":
    main()
