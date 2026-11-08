#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve D25 accepted/refused guards, source identities and checkpoint hashes."""
import hashlib
import json
import os
from pathlib import Path
import tarfile

PHASES = [
    ("r1", "fp16-seed", False),
    ("r2", "late-refusal", False),
    ("r3", "fp16-seed", True),
    ("r3", "fp16-resume", True),
    ("r3", "overflow-seed", True),
    ("r3", "overflow-resume", True),
    ("r3", "inactive-seed", True),
    ("r3", "fp32-control-seed", True),
    ("r3", "bf16-control-seed", True),
    ("r3", "late-refusal", True),
    ("r3", "scale-refusal", True),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.geteuid() == 0
    from guard_distributed_d25_on_host import preflight

    preflight()
    work = Path("/home/nik/bc250work")
    records = {}
    payloads = {}
    for series, label, accepted in PHASES:
        phase = work / "evidence" / ("distributed-d25-" + series) / label
        final = json.loads((phase / "final.json").read_text())
        worker = json.loads((phase / "rank.json").read_text())
        assert (
            final["status"] == worker["status"] == ("passed" if accepted else "failed")
        )
        assert final["kernel_log_unchanged"] and final["registry_unchanged"]
        assert final["docker_containers"] == dict(before=[], after=[])
        state = json.loads((phase / "guard/state.json").read_text())
        ack = json.loads((phase / "guard/ack.json").read_text())
        assert state["status"] == "acknowledged" and all(
            state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
        )
        if accepted:
            assert final["worker"] == worker and final["agent_shutdown_clean"]
        records[series + "/" + label] = dict(final=final, worker=worker)
        if "checkpoint_files" in worker:
            checkpoint = phase / "trainer-output/checkpoint-2"
            for item in worker["checkpoint_files"]:
                assert sha(checkpoint / item["name"]) == item["sha256"]
        if accepted and label in ["fp16-seed", "overflow-seed"]:
            checkpoint = phase / "trainer-output/checkpoint-2"
            values = {
                str(p.relative_to(checkpoint)): dict(
                    bytes=p.stat().st_size, sha256=sha(p)
                )
                for folder in ["optimizer_0", "pytorch_model_fsdp_0"]
                for p in (checkpoint / folder).iterdir()
            }
            assert len(values) == 6
            payloads[label] = values
    roots = [
        work / "evidence" / ("distributed-d25-" + series)
        for series in ["r1", "r2", "r3"]
    ]
    inventory = [
        dict(name=str(p.relative_to(work)), bytes=p.stat().st_size, sha256=sha(p))
        for root in roots
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in [".pt", ".pth", ".bin", ".distcp"]
    ]
    handoff = dict(
        status="passed",
        records=records,
        payload_files=payloads,
        payload_inventory=inventory,
        original_checkpoint_files_unchanged=True,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        kernel=os.uname().release,
        registry_sha256=sha(Path("/opt/bc250-stack/distributed/d01.json")),
        running_containers=[],
    )
    destination = roots[2] / "handoff.json"
    destination.write_text(json.dumps(handoff, indent=2) + "\n")
    archive = work / "evidence/distributed-d25-evidence-r3.tar.gz"
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
            *roots,
            work / "tools/distributed-d25",
            Path("/opt/bc250-stack/distributed/d01.json"),
        ]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for log in (work / "evidence").glob("distributed-d25*controller.log"):
            t.add(log, arcname=str(log).lstrip("/"))
    archive.chmod(0o644)
    print(
        json.dumps(
            dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive))
        )
    )


if __name__ == "__main__":
    main()
