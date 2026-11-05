#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify and export D08 raw metadata; preserve optimizer/scaler checkpoints on board."""
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


def verify_worker(worker: dict, dtype: str, mode: str) -> None:
    assert (
        worker["status"] == "passed"
        and worker["config"]["dtype"] == dtype
        and worker["config"]["mode"] == mode
    )
    assert worker["world"] == 2 and worker["parameter_elements"] == 1051648
    assert (
        worker["local_parameter_elements"]
        == worker["local_AdamW_exp_avg_elements"]
        == 525824
    )
    assert sum(sum(v) for v in worker["layouts"]) == 1051648
    assert len(worker["checks"]) == (12 if dtype == "bf16" else 13)
    assert worker["checkpoint_refusals"] == 6 and len(worker["metrics"]) == 2
    for metric in worker["metrics"]:
        assert metric["maximum_parameter_gradient_relative_L2"] <= (
            0.02 if dtype == "bf16" else 0.004
        )
        assert metric["input_gradient_relative_L2"] <= (
            0.02 if dtype == "bf16" else 0.004
        )
    windows = worker["windows"]
    assert [w["index"] for w in windows] == [0, 1, 2, 3, 4, 5, 6, 6]
    assert [w["local_finite_flags"] for w in windows] == [[True, True]] * 3 + [
        [True, False]
    ] * 2 + [[True, True]] * 3
    assert [w["global_finite"] for w in windows] == [True] * 3 + [False] * 2 + [
        True
    ] * 3
    assert [w["fault"] for w in windows] == [None] * 3 + ["inf", "nan"] + [None] * 3
    scales = (
        [(1, 1)] * 8
        if dtype == "bf16"
        else [(16, 16), (16, 32), (32, 32), (32, 16), (16, 8), (8, 8), (8, 16), (8, 16)]
    )
    assert [(w["scale_before"], w["scale_after"]) for w in windows] == scales
    for window in windows:
        assert len(window["micros"]) == 4
        for i, micro in enumerate(window["micros"]):
            skip = mode == "no-sync" and i < 3
            assert micro["bytes"] == (0 if skip else 4206592)
            assert micro["calls"] == (0 if skip else 8)
            assert micro["gradient_elements"] == (1051648 if skip else 525824)
            low = "torch.bfloat16" if dtype == "bf16" else "torch.float16"
            assert micro["gradient_dtypes"] == [low if skip else "torch.float32"]
    assert worker["training_memory"]["micro_snapshots"] == windows[2]["micros"]
    assert (
        worker["training_memory"]["peak_allocated_bytes"]
        >= worker["training_memory"]["allocated_after_bytes"]
    )


def main():
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d08-r1"
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
        assert state["status"] == "acknowledged"
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
    archive = work / "evidence/distributed-d08-evidence-r1.tar.gz"
    assert not archive.exists()

    def keep(item):
        return (
            None if item.name.endswith(".pt") or "/__pycache__" in item.name else item
        )

    with tarfile.open(archive, "w:gz", compresslevel=1) as t:
        for path in [
            root,
            work / "tools/distributed-d08",
            Path("/opt/bc250-stack/distributed/d01.json"),
            runtime,
        ]:
            t.add(path, arcname=str(path).lstrip("/"), filter=keep)
        for path in (work / "evidence").glob("distributed-d08*controller.log"):
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
