#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export source-bound D42 metadata; only104 rehashes its original raw references."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile
from board_preflight import reservation
from artifact_d35 import load_manifest
from capacity_plan_d42 import descriptor, HOST_REGISTRY_SHA256
from capacity_plan_d41 import CONFIG, CONTROL, parameters

LABELS = ["control", "large-audit"]


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d42-r4"
    rank = 0 if os.uname().nodename == "bc250" else 1
    lock = Path(
        "/opt/bc250-stack/current/.runtime.lock"
        if rank == 0
        else "/opt/bc250-stack/migration-control/.runtime.lock"
    )
    with reservation("d42-final-export"), lock.open("r") as fd:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert (
            hashlib.sha256(
                Path("/opt/bc250-stack/distributed/d01.json").read_bytes()
            ).hexdigest()
            == HOST_REGISTRY_SHA256[rank]
        )
        records = {}
        for label, shape in zip(LABELS, ["control", "large"]):
            phase = root / label
            final = json.loads((phase / "final.json").read_text())
            worker = json.loads((phase / "rank.json").read_text())
            assert (
                final["status"] == worker["status"] == "passed"
                and final["worker"] == worker
            )
            assert (
                final["kernel_log_unchanged"]
                and final["registry_unchanged"]
                and final["agent_shutdown_clean"]
            )
            state = json.loads((phase / "guard/state.json").read_text())
            ack = json.loads((phase / "guard/ack.json").read_text())
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            proof = descriptor(shape)
            if rank == 0:
                path = (
                    work
                    / "evidence"
                    / (
                        "distributed-d35-r1/cpu-reference/reference/manifest.json"
                        if shape == "control"
                        else "distributed-d41-r1/large-oracle/reference/manifest.json"
                    )
                )
                manifest = load_manifest(
                    path,
                    proof["manifest_sha256"],
                    CONTROL if shape == "control" else CONFIG,
                    parameters(CONTROL if shape == "control" else CONFIG),
                    check_sources=False,
                )
                assert manifest == proof["manifest"]
                source = json.loads((phase / "source.json").read_text())
                assert (
                    source["status"] == "stopped"
                    and source["transport"]["failed_requests"]
                    == source["transport"]["rejected_requests"]
                    == 0
                )
            else:
                assert sorted(
                    p.name for p in (phase / "remote-reference").iterdir()
                ) == ["manifest.json"]
            assert not (phase / "actual-gradients").exists()
            records[label] = dict(final=final, worker=worker)
        h = dict(
            status="passed",
            records=records,
            rank=rank,
            original_source_blobs_reverified_on_104=rank == 0,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            kernel=os.uname().release,
            host_MemTotal_bytes=int(
                next(
                    v.split()[1]
                    for v in Path("/proc/meminfo").read_text().splitlines()
                    if v.startswith("MemTotal:")
                )
            )
            * 1024,
            selected_daily=str(Path("/opt/bc250-stack/current").resolve()),
        )
        (root / "handoff.json").write_text(json.dumps(h, indent=2) + "\n")
        archive = work / "evidence/distributed-d42-evidence-r4.tar.gz"
        assert not archive.exists()
        with tarfile.open(archive, "w:gz", compresslevel=1) as t:
            for path in [
                root,
                work / "tools/distributed-d42",
                Path("/opt/bc250-stack/distributed/d01.json"),
                work / "evidence/distributed-d42-r1",
                work / "evidence/distributed-d42-r2",
                work / "evidence/distributed-d42-r3",
                work / "tools/distributed-d42-preliminary-r1",
            ]:
                t.add(
                    path,
                    arcname=str(path).lstrip("/"),
                    filter=lambda m: (
                        None
                        if "__pycache__" in m.name or m.name.endswith(".raw")
                        else m
                    ),
                )
            for path in (work / "evidence").glob("distributed-d42*controller.log"):
                t.add(path, arcname=str(path).lstrip("/"))
        print(
            json.dumps(
                dict(
                    path=str(archive),
                    bytes=archive.stat().st_size,
                    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                )
            )
        )


if __name__ == "__main__":
    main()
