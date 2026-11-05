#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify deployed D09 tools, live recovery and bounded consumer evidence; export metadata."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

PHASES = ["consumer-candidate", "retained-candidate", "consumer-installed"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert os.geteuid() == 0
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d09-r1"
    install = work / "evidence/distributed-d09-tool-install-r1"
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    records = []
    for label in PHASES:
        folder = root / label
        final = json.loads((folder / "final.json").read_text())
        assert final["status"] == "passed" and all(
            final[k]
            for k in [
                "agent_shutdown_clean",
                "kernel_log_unchanged",
                "registry_unchanged",
            ]
        )
        assert final["selected_tools"] == "/opt/bc250-stack/tools/distributed-d09"
        state = json.loads((folder / "guard/state.json").read_text())
        ack = json.loads((folder / "guard/ack.json").read_text())
        assert state["status"] == "acknowledged" and all(
            state[k] == ack[k] for k in ["nonce", "boot_id", "phase"]
        )
        assert final["worker"] == json.loads((folder / "rank.json").read_text())
        assert len(final["worker"]["checks"]) == (
            13 if label == "retained-candidate" else 5
        )
        records.append(dict(label=label, final=final))
    user = json.loads(
        (work / "evidence/distributed-d09-user-cpu/rank.json").read_text()
    )
    assert (
        user["status"] == "passed"
        and user["uid"] == 1000
        and user["torch_cuda_initialized"] is False
        and len(user["checks"]) == 3
    )
    refusals = json.loads((root / "CLI-refusals.json").read_text())
    assert (
        refusals["status"] == "passed"
        and len(refusals["cases"]) == 12
        and refusals["containers_started"] == 0
    )
    assert json.loads((install / "rollback.json").read_text())["status"] == "passed"
    data = json.loads(Path("/opt/bc250-stack/distributed/d01.json").read_text())
    before = json.loads((install / "registry-before.json").read_text())
    assert (
        data["runtime_sha256"] == before["runtime_sha256"]
        and data["release"] == before["release"]
    )
    assert (
        data["agent_exit_fix"] == before["agent_exit_fix"]
        and data["agent_exit_recovery"] == before["agent_exit_recovery"]
    )
    tools = Path(data["launcher_tools"])
    assert tools == Path("/opt/bc250-stack/tools/distributed-d09")
    assert (
        Path("/usr/local/bin/bc250-distributed").resolve()
        == tools / "bc250_distributed.py"
    )
    pins = json.loads(
        (work / "tools/distributed-d09/install-payload/payload-hashes.json").read_text()
    )
    assert all(sha(tools / name) == digest for name, digest in pins.items())
    assert all(
        r["final"]["launcher_sha256"] == pins["bc250_distributed.py"] for r in records
    )
    runtime = Path(data["release"]) / "runtime.json"
    assert sha(runtime) == data["runtime_sha256"]
    summary = dict(
        status="passed",
        records=records,
        user_cpu=user,
        CLI_refusal_count=12,
        installed_payload_hashes=pins,
        runtime=data,
        selected_daily=str(Path("/opt/bc250-stack/current").resolve()),
        kernel=os.uname().release,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        running_containers=[],
    )
    (root / "handoff.json").write_text(json.dumps(summary, indent=2) + "\n")
    archive = work / "evidence/distributed-d09-evidence-r1.tar.gz"
    assert not archive.exists()

    def keep(item):
        return (
            None if item.name.endswith(".pt") or "/__pycache__" in item.name else item
        )

    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for path in [
            root,
            install,
            work / "evidence/distributed-d09-user-cpu",
            work / "tools/distributed-d09",
            tools,
            Path("/opt/bc250-stack/distributed/d01.json"),
            runtime,
            Path("/var/tmp/bc250-d09-fixtures"),
        ]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for name in [
            "distributed_d08_fsdp_amp.py",
            "distributed_d07_amp.py",
            "distributed_d05_train.py",
        ]:
            path = work / "tools/distributed-d08" / name
            t.add(path, arcname=str(path).lstrip("/"))
        adapter = work / "tools/distributed-d05/distributed_d05_bounded_launch.py"
        t.add(adapter, arcname=str(adapter).lstrip("/"))
        for path in (work / "evidence").glob("distributed-d09*controller.log"):
            t.add(path, arcname=str(path).lstrip("/"))
    archive.chmod(0o644)
    print(
        json.dumps(
            dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive))
        )
    )


if __name__ == "__main__":
    main()
