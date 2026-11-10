#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded .104-only old/candidate visibility comparison with independent recovery."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from board_preflight import reservation, preflight

WORK = Path("/home/nik/bc250work")
TOOLS = WORK / "tools/visibility-d44"
ROOT = WORK / "evidence/visibility-d44-r2"
OLD_SHA = "c5167f171eefb6df728eea2c1063f2af1ca1317a03ea8d69ab67f36a700f6106"


def owned():
    return subprocess.check_output(
        ["podman", "ps", "-q", "--filter", "label=bc250.d44=visibility-r1"], text=True
    ).split()


def available():
    return (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(parents=True, exist_ok=False)
    old = Path("/usr/local/bin/bc250-torch")
    assert hashlib.sha256(old.read_bytes()).hexdigest() == OLD_SHA
    assert not any(
        k in os.environ
        for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    )
    result = dict(
        status="running",
        kernel=os.uname().release,
        phases=[],
        sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in TOOLS.glob("*.py")
        },
    )
    state = None
    guard = None
    with reservation("d44-visibility"):
        assert shutil.disk_usage(ROOT).free > 5 * 1024**3
        guarddir = ROOT / "guard"
        with (ROOT / "guard.log").open("w") as guardlog:
            guard = subprocess.Popen(
                [
                    "python3",
                    str(TOOLS / "deadline_guard_d42.py"),
                    "--release",
                    os.uname().release,
                    "--phase",
                    "d44-visibility",
                    "--timeout",
                    "600",
                    "--directory",
                    str(guarddir),
                ],
                stdout=guardlog,
                stderr=subprocess.STDOUT,
            )
        try:
            for _ in range(100):
                if (guarddir / "state.json").exists():
                    break
                time.sleep(0.05)
            state = json.loads((guarddir / "state.json").read_text())
            assert state["status"] == "armed"
            result["boot_id"] = state["boot_id"]
            cases = [
                (
                    "old-negative",
                    old,
                    [
                        "--env",
                        "HIP_VISIBLE_DEVICES=-1",
                        "-c",
                        "import torch; print('IMPORT_RETURNED', flush=True)",
                    ],
                    None,
                ),
                (
                    "candidate-negative",
                    TOOLS / "bc250_torch.py",
                    [
                        "--env",
                        "HIP_VISIBLE_DEVICES=-1",
                        "-c",
                        "raise RuntimeError('WORKER_MUST_NOT_RUN')",
                    ],
                    2,
                ),
                (
                    "candidate-cpu",
                    TOOLS / "bc250_torch.py",
                    [
                        "-c",
                        "import torch; assert not torch.cuda.is_initialized(); print('CPU_IMPORT_UNINITIALIZED', flush=True)",
                    ],
                    0,
                ),
                (
                    "candidate-daily-default",
                    TOOLS / "bc250_torch.py",
                    [str(TOOLS / "visibility_d44_probe.py")],
                    0,
                ),
                (
                    "candidate-daily-zero",
                    TOOLS / "bc250_torch.py",
                    [
                        "--env",
                        "HIP_VISIBLE_DEVICES=0",
                        str(TOOLS / "visibility_d44_probe.py"),
                    ],
                    0,
                ),
                (
                    "candidate-rccl-zero",
                    TOOLS / "bc250_torch.py",
                    [
                        "--runtime-profile",
                        "rccl-single",
                        "--env",
                        "HIP_VISIBLE_DEVICES=0",
                        str(TOOLS / "visibility_d44_probe.py"),
                    ],
                    0,
                ),
            ]
            for name, tool, args, expected in cases:
                before = subprocess.check_output(["dmesg"])
                (ROOT / f"{name}-kernel-before.log").write_bytes(before)
                command = [
                    "python3",
                    str(TOOLS / "visibility_d44_launch.py"),
                    str(tool),
                    "--shm-size",
                    "256m",
                    *args,
                ]
                phase = dict(name=name, command=command, samples=[], containers=[])
                result["phases"].append(phase)
                start = time.monotonic()
                with (ROOT / f"{name}.log").open("w") as log:
                    child = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        while child.poll() is None:
                            ids = owned()
                            phase["containers"] = sorted(
                                set(phase["containers"]) | set(ids)
                            )
                            sample = dict(
                                available_bytes=available(),
                                disk_free_bytes=shutil.disk_usage(ROOT).free,
                            )
                            phase["samples"].append(sample)
                            assert sample["available_bytes"] >= 3 * 1024**3, sample
                            assert sample["disk_free_bytes"] >= 5 * 1024**3, sample
                            assert time.monotonic() - start < 60, "Child deadline"
                            time.sleep(0.1)
                        phase["exit"] = child.returncode
                    finally:
                        for cid in owned():
                            subprocess.run(
                                ["podman", "stop", "--time", "5", cid],
                                check=True,
                                timeout=15,
                            )
                        if child.poll() is None:
                            child.terminate()
                        child.wait(timeout=15)
                phase["seconds"] = time.monotonic() - start
                after = subprocess.check_output(["dmesg"])
                (ROOT / f"{name}-kernel-after.log").write_bytes(after)
                phase["kernel_unchanged"] = before == after
                if expected is not None:
                    assert phase["exit"] == expected, phase
                    assert phase["kernel_unchanged"], name
                if name == "candidate-negative":
                    assert not phase["containers"]
                    text = (ROOT / f"{name}.log").read_text()
                    assert (
                        "recorded ROCr import crash" in text
                        and "WORKER_MUST_NOT_RUN" not in text
                    )
                elif name.startswith("candidate-") and name != "candidate-cpu":
                    payload = json.loads(
                        (ROOT / f"{name}.log").read_text().splitlines()[-1]
                    )
                    assert payload["status"] == "passed" and payload[
                        "architecture"
                    ].startswith("gfx1013")
                    phase["worker"] = payload
                preflight("d44-after-" + name)
            result["status"] = "passed"
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            # Never ACK a still-owned container. Foreign applications are never stopped.
            assert not owned(), "Owned workload remains; recovery stays armed"
            result["postflight"] = preflight("d44-final")
            assert state is not None
            (guarddir / "ack.json").write_text(
                json.dumps({k: state[k] for k in ("boot_id", "nonce", "phase")})
            )
            assert guard.wait(timeout=10) == 0
            result["guard"] = json.loads((guarddir / "state.json").read_text())
            (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
