#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inventory/back up obsolete loose agent files; retire only a verified frozen plan."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

WORK = Path("/home/nik/bc250work")
KEEP_LINKS = {
    "distributed-d04-fsdp-r1",
    "distributed-d01-target-qualification-r1",
    "distributed-d01-recovery",
    "promote_distributed_d01_host105.py",
}
# These absolute paths are still referenced by runtime qualification and rollback.
SENSITIVE = re.compile(
    rb"-----BEGIN .*PRIVATE KEY-----|sshpass\s+-p|(?:password|passwd|token|secret)\s*[:=]\s*[\x22\x27]?[A-Za-z0-9+/=_-]{4,}",
    re.I,
)


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8388608), b""):
            h.update(chunk)
    return h.hexdigest()


def references():
    roots = [
        Path("/etc/init.d"),
        Path("/etc/conf.d"),
        Path("/etc/local.d"),
        Path("/etc/crontabs"),
        Path("/opt/bc250-stack/tools"),
        Path("/opt/bc250-stack/distributed"),
    ]
    result = set()
    for root in roots:
        for p in root.rglob("*"):
            if not p.is_file() or p.stat().st_size > 1048576:
                continue
            data = p.read_bytes()
            if b"\x00" in data:
                continue
            for name in re.findall(rb"/home/nik/bc250work/([^\s\x22\x27;\)]+)", data):
                result.add(name.decode(errors="replace").split("/")[0])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--consolidate-staging", action="store_true")
    p.add_argument("--discard-host-backup", action="store_true")
    p.add_argument("--verified-archive-sha256")
    a = p.parse_args()
    assert os.geteuid() == 0
    assert os.uname().nodename in ("bc250", "bc500")
    host = "104" if os.uname().nodename == "bc250" else "105"
    base = WORK / "evidence" / "top-level-cleanup-20261104"
    base.mkdir(parents=True, exist_ok=True)
    archive = base / "loose-files.tar.gz"
    planfile = base / "plan.json"
    lock = Path(
        "/opt/bc250-stack/current/.runtime.lock"
        if host == "104"
        else "/opt/bc250-stack/migration-control/.runtime.lock"
    )
    with lock.open("r") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        if a.discard_host_backup:
            result_path = base / "result.json"
            result = json.loads(result_path.read_text())
            plan = json.loads(planfile.read_text())
            assert result["status"] == "passed"
            assert (
                a.verified_archive_sha256 == plan["archive_sha256"] == digest(archive)
            )
            archive.unlink()
            result["backup_location"] = (
                "verified workstation archive; host transfer retired"
            )
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            return
        if a.consolidate_staging:
            result_path = base / "result.json"
            result = json.loads(result_path.read_text())
            assert result["status"] == "passed" and "staging_moves" not in result
            moves = []
            for name, relative in [
                ("distributed-d01-tools", "tools/distributed-d01/installer-inputs"),
                ("distributed-d01-import", "evidence/distributed-d01/transfer-import"),
                ("distributed-d01-export", "evidence/distributed-d01/transfer-export"),
            ]:
                source = WORK / name
                if source.is_dir() and not source.is_symlink():
                    target = WORK / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    assert not target.exists()
                    source.rename(target)
                    moves.append(dict(old=str(source), new=str(target)))
            result["staging_moves"] = moves
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(moves))
            return
        if not a.apply:
            assert not archive.exists() and not planfile.exists()
            protected = references() | KEEP_LINKS
            rows = []
            with tarfile.open(archive, "w:gz", dereference=False) as tar:
                for item in sorted(WORK.iterdir()):
                    if item.name in protected or not (
                        item.is_file() or item.is_symlink()
                    ):
                        continue
                    if item.is_symlink():
                        rows.append(dict(name=item.name, link=os.readlink(item)))
                        continue
                    sensitive = item.stat().st_size < 1048576 and bool(
                        SENSITIVE.search(item.read_bytes())
                    )
                    row = dict(
                        name=item.name,
                        sha256=digest(item),
                        bytes=item.stat().st_size,
                        archived=not sensitive,
                    )
                    if not sensitive:
                        tar.add(item, arcname=item.name)
                    rows.append(row)
            planfile.write_text(
                json.dumps(
                    dict(
                        host=host,
                        protected=sorted(protected),
                        rows=rows,
                        archive_sha256=digest(archive),
                    ),
                    indent=2,
                )
                + "\n"
            )
            archive.chmod(0o644)
            planfile.chmod(0o644)
            print(
                json.dumps(
                    dict(
                        host=host,
                        files=len(rows),
                        archive_sha256=digest(archive),
                        bytes=archive.stat().st_size,
                    )
                )
            )
            return
        plan = json.loads(planfile.read_text())
        assert a.verified_archive_sha256 == plan["archive_sha256"] == digest(archive)
        for row in plan["rows"]:
            path = WORK / row["name"]
            if "link" in row:
                assert path.is_symlink() and os.readlink(path) == row["link"]
            else:
                assert not path.is_symlink() and digest(path) == row["sha256"]
        for row in plan["rows"]:
            (WORK / row["name"]).unlink()
        result = dict(
            status="passed",
            removed=len(plan["rows"]),
            removed_logical_bytes=sum(x.get("bytes", 0) for x in plan["rows"]),
            retained_references=plan["protected"],
        )
        (base / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
