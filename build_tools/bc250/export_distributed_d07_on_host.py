#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify and export D07 raw metadata; preserve optimizer/scaler checkpoints on board."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

PHASES = [
    (dtype, mode, f"{dtype}-{mode}-r1")
    for dtype in ["bf16", "fp16"]
    for mode in ["no-sync", "sync"]
]


def verify_worker(worker, dtype, mode):
    assert worker["status"] == "passed"
    assert worker["config"]["dtype"] == dtype and worker["config"]["mode"] == mode
    assert worker["world"] == 2 and worker["parameter_elements"] == 1051648
    assert worker["gradient_dtype"] == "torch.float32"
    assert len(worker["checks"]) == (11 if dtype == "bf16" else 12)
    assert worker["checkpoint_refusals"] == 6
    assert len(worker["metrics"]) == 2
    for metric in worker["metrics"]:
        assert metric["maximum_parameter_gradient_relative_L2"] <= (
            0.02 if dtype == "bf16" else 0.004
        )
        assert metric["input_gradient_relative_L2"] <= (
            0.02 if dtype == "bf16" else 0.004
        )
    windows = worker["windows"]
    assert [w["index"] for w in windows] == [0, 1, 2, 3, 4, 5, 5]
    assert [w["finite"] for w in windows] == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    assert [w["fault"] for w in windows] == [None, None, "inf", "nan", None, None, None]
    expected_scales = (
        [(1, 1)] * 7
        if dtype == "bf16"
        else [(16, 16), (16, 32), (32, 16), (16, 8), (8, 8), (8, 16), (8, 16)]
    )
    assert [(w["scale_before"], w["scale_after"]) for w in windows] == expected_scales
    for window in windows:
        assert len(window["micros"]) == 4
        for index, micro in enumerate(window["micros"]):
            active = mode == "sync" or index == 3
            assert micro["bytes"] == (1051648 * 4 if active else 0)
            assert bool(micro["calls"]) == active


def main():
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d07-r1"
    records, checkpoints = [], []
    boots, registries, sources = set(), set(), set()
    for dtype, mode, label in PHASES:
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
        worker = json.loads((folder / "rank.json").read_text())
        assert worker == final["worker"]
        verify_worker(worker, dtype, mode)
        state = json.loads((folder / "guard/state.json").read_text())
        ack = json.loads((folder / "guard/ack.json").read_text())
        assert all(state[k] == ack[k] for k in ["nonce", "phase", "boot_id"])
        assert (folder / "kernel-before.log").read_bytes() == (
            folder / "kernel-after.log"
        ).read_bytes()
        boots.add(final["boot_id"])
        registries.add(final["registry_sha256"])
        sources.add(worker["script_sha256"])
        for path in (folder / "checkpoint").glob("*.pt"):
            checkpoints.append(dict(path=str(path), bytes=path.stat().st_size))
        records.append(dict(label=label, final=final))
    assert len(boots) == len(registries) == len(sources) == 1
    info = json.loads(Path("/opt/bc250-stack/distributed/d01.json").read_text())
    runtime = Path(info["release"]) / "runtime.json"
    assert sha(runtime) == info["runtime_sha256"]
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
    archive = work / "evidence/distributed-d07-evidence-r1.tar.gz"
    assert not archive.exists()

    def keep(item):
        return (
            None if item.name.endswith(".pt") or "/__pycache__" in item.name else item
        )

    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for path in [
            root,
            work / "tools/distributed-d07",
            Path("/opt/bc250-stack/distributed/d01.json"),
            runtime,
        ]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for path in (work / "evidence").glob("distributed-d07*controller.log"):
            t.add(path, arcname=str(path).lstrip("/"))
    archive.chmod(0o644)
    print(
        json.dumps(
            dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive))
        )
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
