#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve accepted and failed D24 guards, checkpoint hashes and source evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

PHASES = [
    ("bf16-seed", "distributed-d24", True, 14),
    ("bf16-resume", "distributed-d24", True, 7),
    ("inactive-seed", "distributed-d24", True, 4),
    ("fp32-control-seed", "distributed-d24", True, 4),
    ("nan-refusal", "distributed-d24", True, 3),
]


def main():
    assert (
        os.geteuid() == 0
        and not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    )
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d24-r1"
    records = {}
    for label, tools, accepted, groups in PHASES:
        phase = root / label
        final = json.loads((phase / "final.json").read_text())
        worker = json.loads((phase / "rank.json").read_text())
        assert (
            final["status"] == ("passed" if accepted else "failed")
            and worker["status"] == final["status"]
        )
        assert (
            len(worker["checks"]) == groups
            and final["kernel_log_unchanged"]
            and final["registry_unchanged"]
        )
        if accepted:
            assert final["worker"] == worker and final["agent_shutdown_clean"]
        state = json.loads((phase / "guard/state.json").read_text())
        ack = json.loads((phase / "guard/ack.json").read_text())
        assert state["status"] == "acknowledged" and all(
            state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
        )
        records[label] = dict(final=final, worker=worker)
    checkpoint = root / "bf16-seed/trainer-output/checkpoint-2"
    original = records["bf16-seed"]["worker"]["checkpoint_files"]
    for item in original:
        assert (
            hashlib.sha256((checkpoint / item["name"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    payload_files = {
        str(f.relative_to(checkpoint)): dict(
            bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest()
        )
        for folder in ["optimizer_0", "pytorch_model_fsdp_0"]
        for f in (checkpoint / folder).iterdir()
    }
    assert len(payload_files) == 6
    payload_inventory = [
        dict(
            name=str(p.relative_to(root)),
            bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in [".pt", ".pth", ".bin", ".distcp"]
    ]
    h = dict(
        status="passed",
        records=records,
        payload_files=payload_files,
        original_checkpoint_files_unchanged=True,
        payload_inventory=payload_inventory,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        kernel=os.uname().release,
        registry_sha256=hashlib.sha256(
            Path("/opt/bc250-stack/distributed/d01.json").read_bytes()
        ).hexdigest(),
        running_containers=[],
    )
    (root / "handoff.json").write_text(json.dumps(h, indent=2) + "\n")
    archive = work / "evidence/distributed-d24-evidence-r1.tar.gz"
    assert not archive.exists()

    def keep(info):
        return (
            None
            if "/__pycache__" in info.name
            or Path(info.name).suffix in [".pt", ".pth", ".bin", ".distcp"]
            else info
        )

    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for path in [
            root,
            Path("/opt/bc250-stack/distributed/d01.json"),
            *[work / "tools" / name for name in sorted({x[1] for x in PHASES})],
        ]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for log in (work / "evidence").glob("distributed-d24*controller.log"):
            t.add(log, arcname=str(log).lstrip("/"))
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
