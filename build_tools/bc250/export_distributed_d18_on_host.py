#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify native evaluation guards and export reproducible sources/raw evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

LABELS = [
    f"{model}-{precision}-eval"
    for model in ["bert", "gpt2"]
    for precision in ["fp32", "bf16", "fp16"]
]


def main():
    assert (
        os.geteuid() == 0
        and not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    )
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d18-r1"
    records = {
        label: json.loads((root / label / "final.json").read_text()) for label in LABELS
    }
    for label, final in records.items():
        assert final["status"] == "passed" and all(
            final[k]
            for k in [
                "agent_shutdown_clean",
                "kernel_log_unchanged",
                "registry_unchanged",
            ]
        )
        assert final["worker"] == json.loads((root / label / "rank.json").read_text())
        assert (
            len(final["worker"]["checks"]) == 9
            and len(final["worker"]["CPU_contract_checks"]) == 4
        )
        state = json.loads((root / label / "guard/state.json").read_text())
        ack = json.loads((root / label / "guard/ack.json").read_text())
        assert state["status"] == "acknowledged" and all(
            state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
        )
    assert all(v["exit"] == 0 for v in json.loads((root / "sequence.json").read_text()))
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    assert (
        json.loads(registry.read_text())["launcher_tools"]
        == "/opt/bc250-stack/tools/distributed-d09"
    )
    handoff = dict(
        status="passed",
        records=records,
        kernel=os.uname().release,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
        running_containers=[],
    )
    (root / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
    archive = work / "evidence/distributed-d18-evidence-r1.tar.gz"
    assert not archive.exists()

    def keep(item):
        return None if "/__pycache__" in item.name else item

    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for path in [root, work / "tools/distributed-d18", registry]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for path in (work / "evidence").glob("distributed-d18*controller.log"):
            t.add(path, arcname=str(path).lstrip("/"))
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
