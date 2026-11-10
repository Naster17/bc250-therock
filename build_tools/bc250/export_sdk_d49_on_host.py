#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export D49 raw tests, source bytes, transactions and final installed identities."""
import json
import os
from pathlib import Path
import tarfile

from board_preflight import reservation, preflight
from stage_sdk_d49_on_host import (
    BASE,
    ROOT,
    OLD,
    WORK,
    SDK,
    NEW_SHA,
    OLD_SHA,
    sha,
    write,
    identity_tree,
)


def main():
    with reservation("d49-export"):
        controller = json.loads(
            (WORK / "evidence/sdk-d49-controller/result.json").read_text()
        )
        assert controller["status"] == "passed"
        assert (BASE / "current").resolve() == ROOT / "profiles/daily"
        assert (
            sha(ROOT / "rocm" / SDK) == NEW_SHA and sha(OLD / "rocm" / SDK) == OLD_SHA
        )
        preservation = json.loads((ROOT / "preservation.json").read_text())
        assert identity_tree(OLD / "rocm") == preservation["old"]
        assert identity_tree(ROOT / "rocm") == preservation["new"]
        stage = json.loads((ROOT / "stage.json").read_text())
        for name, digest in stage["profiles"].items():
            assert sha(ROOT / "profiles" / name / "runtime.json") == digest
        assert sha(Path("/usr/local/bin/bc250-torch")) == sha(
            BASE / "tools/distributed-d09/bc250_torch.py"
        )
        for phase in ("activation-r1", "rollback-r1", "final-r1"):
            record = json.loads(
                (WORK / "evidence" / ("sdk-d49-" + phase) / "result.json").read_text()
            )
            assert (
                record["status"] == "passed"
                and record["guard"]["status"] == "acknowledged"
            )
        for profile in ("daily", "rccl-single"):
            paths = (BASE / "current", BASE / "profiles/rccl-single")
            assert (
                ROOT / "profiles" / profile / "native-overrides.json"
            ).read_bytes() == (
                ROOT / "profiles/daily/native-overrides.json"
            ).read_bytes()
        summary = dict(
            status="passed",
            sdk_sha256=NEW_SHA,
            old_sdk_sha256=OLD_SHA,
            helper_sha256=sha(Path("/usr/local/bin/bc250-torch")),
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            kernel=os.uname().release,
            pointers={
                str(p): os.readlink(p)
                for p in [
                    BASE / "current",
                    BASE / "profiles/rccl-single",
                    BASE / "native-cohort",
                    *[
                        Path("/usr/local/bin/bc250-" + n)
                        for n in ("ck", "migraphx", "ort")
                    ],
                ]
            },
            distributed=json.loads((BASE / "distributed/d01.json").read_text()),
            preflight=preflight("d49-export-final"),
            root_preserved=True,
            host105_accessed=False,
        )
        folder = WORK / "evidence/sdk-d49-export"
        folder.mkdir(exist_ok=False)
        write(folder / "final.json", summary)
        archive = WORK / "evidence/sdk-d49-evidence-r1.tar.gz"
        assert not archive.exists()
        with tarfile.open(archive, "w:gz") as output:
            for name in (
                "sdk-d49-controller",
                "sdk-d49-install",
                "sdk-d49-activation-r1",
                "sdk-d49-rollback-r1",
                "sdk-d49-final-r1",
                "sdk-d49-export",
            ):
                output.add(WORK / "evidence" / name, arcname="evidence/" + name)
            output.add(WORK / "tools/sdk-d49", arcname="tools")
            for name in ("stage.json", "preservation.json"):
                output.add(ROOT / name, arcname="cohort/" + name)
            for profile in ("daily", "rccl-single"):
                for name in ("runtime.json", "native-overrides.json"):
                    output.add(
                        ROOT / "profiles" / profile / name,
                        arcname="cohort/profiles/" + profile + "/" + name,
                    )
        print(
            json.dumps(
                dict(
                    path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size
                )
            )
        )


if __name__ == "__main__":
    main()
