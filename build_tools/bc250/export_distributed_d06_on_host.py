#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify/export D06 metadata and logs while retaining owned checkpoints on board."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from run_distributed_d06_on_host import PHASES


def main():
    assert os.geteuid() == 0
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d06-r1"
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    records, checkpoints = [], []
    boots, registries, sources = set(), set(), set()
    for case, mode, label in PHASES:
        folder = root / label
        final = json.loads((folder / "final.json").read_text())
        assert final["status"] == "passed"
        assert all(
            final[k]
            for k in [
                "agent_shutdown_clean",
                "kernel_log_unchanged",
                "registry_unchanged",
            ]
        )
        worker = final["worker"]
        assert worker == json.loads((folder / "rank.json").read_text())
        assert worker["mode"] == mode and worker["status"] == "passed"
        assert worker["config"]["global_batch"] == (8 if case == "small" else 32)
        assert len(worker["checks"]) == 8 and worker["checkpoint_refusals"] == 6
        assert len(worker["step_seconds"]) == 5 and len(worker["traffic_steps"]) == 11
        expected = worker["gradient_bytes"] * (
            worker["accumulation"] if mode == "sync" else int(mode != "single")
        )
        for step in worker["traffic_steps"]:
            assert step["bytes"] == expected
            assert len(step["microbatches"]) == worker["accumulation"]
            for i, micro in enumerate(step["microbatches"]):
                active = (
                    mode == "sync"
                    or mode == "no-sync"
                    and i + 1 == worker["accumulation"]
                )
                assert micro["bytes"] == (worker["gradient_bytes"] if active else 0)
                assert bool(micro["calls"]) == active
        state = json.loads((folder / "guard/state.json").read_text())
        ack = json.loads((folder / "guard/ack.json").read_text())
        assert all(state[k] == ack[k] for k in ["nonce", "phase", "boot_id"])
        boots.add(final["boot_id"])
        registries.add(final["registry_sha256"])
        sources.add(worker["script_sha256"])
        for path in (folder / "checkpoint").glob("*.pt"):
            checkpoints.append(dict(path=str(path), bytes=path.stat().st_size))
        records.append(dict(label=label, final=final))
    assert len(boots) == len(registries) == len(sources) == 1
    info = json.loads(Path("/opt/bc250-stack/distributed/d01.json").read_text())
    runtime = Path(info["release"]) / "runtime.json"
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == info["runtime_sha256"]
    summary = dict(
        status="passed",
        records=records,
        checkpoints_retained=checkpoints,
        kernel=os.uname().release,
        boot_id=next(iter(boots)),
        selected_daily=str(Path("/opt/bc250-stack/current").resolve()),
        distributed=info,
        running_containers=[],
    )
    (root / "handoff.json").write_text(json.dumps(summary, indent=2) + "\n")
    archive = work / "evidence/distributed-d06-evidence-r1.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz", compresslevel=1) as t:

        def keep(item):
            return (
                None
                if item.name.endswith(".pt") or "/__pycache__" in item.name
                else item
            )

        for path in [
            root,
            work / "tools/distributed-d06",
            Path("/opt/bc250-stack/distributed/d01.json"),
            runtime,
        ]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for path in (work / "evidence").glob("distributed-d06*controller.log"):
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
