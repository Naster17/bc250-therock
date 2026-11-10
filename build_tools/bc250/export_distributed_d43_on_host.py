#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export D43 metadata under cooperative reservation; no raw reference reads."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
from board_preflight import reservation
from capacity_plan_d42 import HOST_REGISTRY_SHA256
from qualification_d43 import proof

LABELS = ["control", "large-training"]


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d43-r3"
    rank = 0 if os.uname().nodename == "bc250" else 1
    with reservation("d43-final-export") as idle:
        registry = Path("/opt/bc250-stack/distributed/d01.json")
        assert (
            hashlib.sha256(registry.read_bytes()).hexdigest()
            == HOST_REGISTRY_SHA256[rank]
        )
        for label, shape in zip(LABELS, ["control", "large"]):
            phase = root / label
            f = json.loads((phase / "final.json").read_text())
            w = json.loads((phase / "rank.json").read_text())
            assert f["status"] == w["status"] == "passed" and f["worker"] == w
            assert (
                f["agent_shutdown_clean"]
                and f["kernel_log_unchanged"]
                and f["registry_unchanged"]
            )
            assert w["qualification_sha256"] == proof(shape, rank)["sha256"]
            state = json.loads((phase / "guard/state.json").read_text())
            ack = json.loads((phase / "guard/ack.json").read_text())
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
        handoff = dict(
            status="passed",
            rank=rank,
            idle=idle,
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
        (root / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
        archive = work / "evidence/distributed-d43-evidence-r3.tar.gz"
        assert not archive.exists()
        with tarfile.open(archive, "w:gz", compresslevel=1) as t:
            for path in [
                root,
                work / "tools/distributed-d43",
                work / "evidence/distributed-d43-r1",
                work / "evidence/distributed-d43-r2",
                registry,
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
            for path in (work / "evidence").glob("distributed-d43*controller.log"):
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
