#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export D05 raw evidence without model checkpoint payloads; keep those on board."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d05-r1"
    labels = [
        "single-a",
        "ddp-a",
        "fsdp-a",
        "fsdp-b",
        "ddp-b",
        "single-b",
        "ddp-fixed",
        "fsdp-fixed",
        "ddp-installed",
    ]
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    acknowledged = []
    checkpoints = []
    for label in labels:
        p = root / label
        result = json.loads((p / "final.json").read_text())
        assert (
            result["status"] == "passed"
            and result["kernel_log_unchanged"]
            and result["registry_unchanged"]
        )
        if label.endswith(("fixed", "installed")):
            assert result["agent_shutdown_clean"]
        state = json.loads((p / "guard/state.json").read_text())
        ack = json.loads((p / "guard/ack.json").read_text())
        assert all(state[k] == ack[k] for k in ["nonce", "phase", "boot_id"])
        acknowledged.append(label)
        for path in (p / "checkpoint").glob("*.pt"):
            checkpoints.append(dict(path=str(path), bytes=path.stat().st_size))
    user = json.loads(
        (work / "evidence/distributed-d05-user-cpu/rank.json").read_text()
    )
    assert (
        user["status"] == "passed"
        and user["HIP_initialized"] is False
        and user["uid"] == 1000
    )
    info = json.loads(Path("/opt/bc250-stack/distributed/d01.json").read_text())
    assert info["launcher_tools"] == "/opt/bc250-stack/tools/distributed-d05"
    runtime = Path(info["release"]) / "runtime.json"
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == info["runtime_sha256"]
    summary = dict(
        status="passed",
        acknowledged_guards=acknowledged,
        checkpoints_retained=checkpoints,
        distributed=info,
        selected_daily=str(Path("/opt/bc250-stack/current").resolve()),
        kernel=os.uname().release,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        containers=[],
    )
    (root / "handoff.json").write_text(json.dumps(summary, indent=2) + "\n")
    archive = work / "evidence/distributed-d05-evidence-r1.tar.gz"
    assert not archive.exists()
    paths = [
        root,
        work / "evidence/distributed-d05-user-cpu",
        work / "evidence/distributed-d05-tool-install-r1",
        work / "tools/distributed-d05",
        Path("/opt/bc250-stack/tools/distributed-d05"),
    ]
    if os.uname().nodename == "bc250":
        paths += [work / "evidence/distributed-d05-exit-cpu-r2"]
    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for p in paths:
            t.add(
                p,
                arcname=str(p).lstrip("/"),
                filter=lambda item: None if item.name.endswith(".pt") else item,
            )
        for p in (work / "evidence").glob("distributed-d05*controller.log"):
            t.add(p, arcname=str(p).lstrip("/"))
        for p in [Path("/opt/bc250-stack/distributed/d01.json"), runtime]:
            t.add(p, arcname=str(p).lstrip("/"))
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
