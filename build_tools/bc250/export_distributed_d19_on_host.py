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
] + ["bertwide-fp32-eval", "gpt2wide-fp32-eval"]


def main():
    assert (
        os.geteuid() == 0
        and not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    )
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d19-r3"
    records = {
        label: json.loads((root / label / "final.json").read_text()) for label in LABELS
    }
    for label, final in records.items():
        phase_root = root
        assert final["status"] == "passed" and all(
            final[k]
            for k in [
                "agent_shutdown_clean",
                "kernel_log_unchanged",
                "registry_unchanged",
            ]
        )
        assert final["worker"] == json.loads(
            (phase_root / label / "rank.json").read_text()
        )
        assert (
            len(final["worker"]["checks"]) == 11
            and len(final["worker"]["CPU_contract_checks"]) == 5
        )
        state = json.loads((phase_root / label / "guard/state.json").read_text())
        ack = json.loads((phase_root / label / "guard/ack.json").read_text())
        assert state["status"] == "acknowledged" and all(
            state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
        )
    assert all(v["exit"] == 0 for v in json.loads((root / "sequence.json").read_text()))
    failed = json.loads(
        (work / "evidence/distributed-d19-r1/bertwide-bf16-eval/final.json").read_text()
    )
    assert (
        failed["status"] == "failed"
        and failed["kernel_log_unchanged"]
        and failed["registry_unchanged"]
    )
    failed_guard = json.loads(
        (
            work / "evidence/distributed-d19-r1/bertwide-bf16-eval/guard/state.json"
        ).read_text()
    )
    assert failed_guard["status"] == "acknowledged"
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    assert (
        json.loads(registry.read_text())["launcher_tools"]
        == "/opt/bc250-stack/tools/distributed-d09"
    )
    handoff = dict(
        status="passed",
        records=records,
        failed_bf16_wide=failed,
        kernel=os.uname().release,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
        running_containers=[],
    )
    (root / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
    archive = work / "evidence/distributed-d19-evidence-r3.tar.gz"
    assert not archive.exists()

    def keep(item):
        return None if "/__pycache__" in item.name else item

    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for path in [
            root,
            work / "evidence/distributed-d19-r1",
            work / "evidence/distributed-d19-r2",
            work / "tools/distributed-d19",
            work / "tools/distributed-d19-r2",
            work / "tools/distributed-d19-r3",
            registry,
        ]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for path in (work / "evidence").glob("distributed-d19*controller.log"):
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
