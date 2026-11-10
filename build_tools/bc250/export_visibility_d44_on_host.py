#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export only owned D44 sources/results, including the failed adapter attempt."""
import hashlib
import json
import os
from pathlib import Path
import tarfile

from board_preflight import reservation

WORK = Path("/home/nik/bc250work")


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    with reservation("d44-export") as admission:
        proof = json.loads(
            (WORK / "evidence/visibility-d44-r2/result.json").read_text()
        )
        install = json.loads(
            (WORK / "evidence/visibility-d44-install-r1/result.json").read_text()
        )
        assert proof["status"] == install["status"] == "passed"
        assert proof["guard"]["status"] == "acknowledged"
        assert (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == proof["boot_id"]
        )
        targets = {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in install["targets"]
        }
        assert set(targets.values()) == {install["source_sha256"]}
        record = dict(
            status="passed",
            preflight=admission,
            targets=targets,
            kernel=os.uname().release,
            boot_id=proof["boot_id"],
            prefix=str(Path("/opt/bc250-stack/current").resolve()),
            registry_sha256=hashlib.sha256(
                Path("/opt/bc250-stack/distributed/d01.json").read_bytes()
            ).hexdigest(),
        )
        (WORK / "evidence/visibility-d44-install-r1/export.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        archive = WORK / "evidence/visibility-d44-evidence-r1.tar.gz"
        assert not archive.exists()
        with tarfile.open(archive, "w:gz") as output:
            for relative in [
                "tools/visibility-d44",
                "evidence/visibility-d44-r1",
                "evidence/visibility-d44-r2",
                "evidence/visibility-d44-install-r1",
                "evidence/visibility-d44-controller.log",
                "evidence/visibility-d44-r2-controller.log",
                "evidence/visibility-d44-install-controller.log",
            ]:
                output.add(WORK / relative, arcname=relative)
        archive.chmod(0o644)
        print(
            json.dumps(
                dict(
                    path=str(archive),
                    size=archive.stat().st_size,
                    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                )
            )
        )


if __name__ == "__main__":
    main()
