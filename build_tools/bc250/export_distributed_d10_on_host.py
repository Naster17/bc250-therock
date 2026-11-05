#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D10 fresh-process continuity and preserve raw diagnostics without .pt data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

LABELS = [f"{c}-{a}" for c in ["ddp", "fsdp"] for a in ["seed", "resume"]]


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d10-r2"
    records = {
        label: json.loads((root / label / "final.json").read_text()) for label in LABELS
    }
    checkpoints = []
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
        state = json.loads((root / label / "guard/state.json").read_text())
        ack = json.loads((root / label / "guard/ack.json").read_text())
        assert state["status"] == "acknowledged" and all(
            state[k] == ack[k] for k in ["nonce", "boot_id", "phase"]
        )
        worker = final["worker"]
        assert len(worker["checks"]) == (4 if label.endswith("seed") else 8)
        if label.endswith("resume"):
            assert worker["checkpoint_refusals"] == 11
    for case in ["ddp", "fsdp"]:
        seed, resume = [records[case + "-" + a] for a in ["seed", "resume"]]
        s, r = seed["worker"], resume["worker"]
        assert seed["container_id"] != resume["container_id"]
        assert (
            r["source_birth"] == s["birth"]
            and s["birth"]["uuid"] != r["birth"]["uuid"]
            and s["birth"]["start_ticks"] != r["birth"]["start_ticks"]
        )
        assert s["cohort"] == r["cohort"] and s["config"] == r["config"]
        assert (
            s["transcript"][3:] == r["transcript"]
            and s["final_state_digest"] == r["final_state_digest"]
        )
        for path in (root / (case + "-seed")).rglob("*.pt"):
            checkpoints.append(dict(path=str(path), bytes=path.stat().st_size))
    for label in ["ddp-seed", "ddp-resume"]:
        p = work / "evidence/distributed-d10-r1" / label
        assert (
            json.loads((p / "guard/state.json").read_text())["status"] == "acknowledged"
        )
    assert (
        json.loads(
            (work / "evidence/distributed-d10-r1/ddp-resume/final.json").read_text()
        )["status"]
        == "failed"
    )
    assert json.loads((root / "sequence.json").read_text())[-1]["exit"] == 0
    info = json.loads(Path("/opt/bc250-stack/distributed/d01.json").read_text())
    assert info["launcher_tools"] == "/opt/bc250-stack/tools/distributed-d09"
    summary = dict(
        status="passed",
        records=records,
        valid_checkpoints_retained=checkpoints,
        registry_sha256=hashlib.sha256(
            Path("/opt/bc250-stack/distributed/d01.json").read_bytes()
        ).hexdigest(),
        kernel=os.uname().release,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        running_containers=[],
    )
    (root / "handoff.json").write_text(json.dumps(summary, indent=2) + "\n")
    archive = work / "evidence/distributed-d10-evidence-r2.tar.gz"
    assert not archive.exists()

    def keep(item):
        return (
            None if item.name.endswith(".pt") or "/__pycache__" in item.name else item
        )

    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for path in [
            root,
            work / "evidence/distributed-d10-r1",
            work / "tools/distributed-d10",
            Path("/opt/bc250-stack/distributed/d01.json"),
        ]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for path in (work / "evidence").glob("distributed-d10*controller.log"):
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
