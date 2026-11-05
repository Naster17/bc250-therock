#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D20 recovery records and preserve sources/identity/diagnostic evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

PHASES = [
    ("r1", "bertwide-bf16-diagnose", "distributed-d20", 6),
    ("r2", "bertwide-bf16-fp32-head", "distributed-d20-r2", 7),
    ("r3", "bertwide-bf16-scope", "distributed-d20-r3", 5),
]


def main():
    assert (
        os.geteuid() == 0
        and not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    )
    work = Path("/home/nik/bc250work")
    records = {}
    for series, label, tools, groups in PHASES:
        root = work / "evidence" / ("distributed-d20-" + series) / label
        final = json.loads((root / "final.json").read_text())
        worker = json.loads((root / "rank.json").read_text())
        assert (
            final["status"] == "passed"
            and final["worker"] == worker
            and worker["status"] == "passed"
        )
        assert len(worker["checks"]) == groups
        if series != "r3":
            assert len(worker["digest_contracts"]) == 3
        if series == "r2":
            assert len(worker["head_contracts"]) == 4
        assert all(
            final[k]
            for k in [
                "agent_shutdown_clean",
                "kernel_log_unchanged",
                "registry_unchanged",
            ]
        )
        state = json.loads((root / "guard/state.json").read_text())
        ack = json.loads((root / "guard/ack.json").read_text())
        assert state["status"] == "acknowledged" and all(
            state[k] == ack[k] for k in ["nonce", "boot_id", "phase"]
        )
        records[series] = final
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    handoff = dict(
        status="passed",
        records=records,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        kernel=os.uname().release,
        registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
        running_containers=[],
    )
    output = work / "evidence/distributed-d20-handoff-r3.json"
    assert not output.exists()
    output.write_text(json.dumps(handoff, indent=2) + "\n")
    archive = work / "evidence/distributed-d20-evidence-r3.tar.gz"
    assert not archive.exists()
    paths = [
        output,
        registry,
        Path("/opt/bc250-stack/tools/distributed-d09/torch-identity.json"),
        Path("/opt/bc250-stack/current/python-consumers/packages.json"),
    ]
    for series, label, tools, groups in PHASES:
        paths.extend(
            [
                work / "evidence" / ("distributed-d20-" + series),
                work / "tools" / tools,
                work / "evidence" / ("distributed-d20-" + series + "-controller.log"),
            ]
        )
    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for path in paths:
            t.add(
                path,
                arcname=str(path).lstrip("/"),
                filter=lambda info: None if "/__pycache__" in info.name else info,
            )
    archive.chmod(0o644)
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
