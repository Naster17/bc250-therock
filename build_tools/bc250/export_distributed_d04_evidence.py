#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Archive owned two-host rollout evidence without images, model data or user apps."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    assert os.geteuid() == 0
    rank = 0 if os.uname().nodename == "bc250" else 1
    root = Path("/home/nik/bc250work")
    names = [
        "distributed-d01-20261104",
        "distributed-d02-rccl-r1",
        "distributed-d03-cli-r1",
        "distributed-d03-cli-r2",
        "distributed-d03-cli-r3",
        "distributed-d04-fsdp-r1",
    ]
    if rank == 1:
        names += [
            "distributed-d01-target-qualification-r1",
            "distributed-d01-ordinary",
            "distributed-d01-ordinary-r2",
            "distributed-d01-recovery",
        ]
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    acknowledged = []
    for name in names:
        folder = root / name
        assert folder.is_dir(), folder
        for state in folder.rglob("state.json"):
            data = json.loads(state.read_text())
            if "nonce" not in data or "boot_id" not in data:
                continue
            ack = json.loads(state.with_name("ack.json").read_text())
            assert all(ack[k] == data[k] for k in ["nonce", "boot_id", "phase"])
            acknowledged.append(str(state.parent))
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    record = json.loads(registry.read_text())
    runtime = Path(record["release"]) / "runtime.json"
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == record["runtime_sha256"]
    selected = Path("/opt/bc250-stack/current").resolve()
    report = root / "distributed-d04-handoff.json"
    report.write_text(
        json.dumps(
            dict(
                status="passed",
                rank=rank,
                selected=str(selected),
                kernel=os.uname().release,
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                containers=[],
                acknowledged_guards=acknowledged,
                distributed=record,
            ),
            indent=2,
        )
        + "\n"
    )
    archive = root / "distributed-d04-evidence-r2.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz", dereference=False) as tar:
        for name in [*names, report.name]:
            tar.add(root / name, arcname=name)
        for path in sorted(root.glob("distributed-d*-controller.log")):
            tar.add(path, arcname=path.name)
        for path in [registry, runtime, selected / "runtime.json"]:
            tar.add(path, arcname="configuration" + str(path))
        tar.add(
            Path("/opt/bc250-stack/tools/distributed-d01"),
            arcname="installed-tools",
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
