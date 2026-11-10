#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Public installed SDK checks, controlled rollback, bounded independent recovery."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from board_preflight import reservation, preflight
from stage_sdk_d49_on_host import BASE, WORK, OLD_SHA, NEW_SHA, SDK, sha, write

TOOLS = WORK / "tools/sdk-d49"


def owned():
    return subprocess.check_output(
        ["podman", "ps", "-q", "--filter", "label=bc250.d49=test"],
        text=True,
        timeout=10,
    ).split()


def resources():
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return dict(available_bytes=available, disk_free_bytes=shutil.disk_usage(WORK).free)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["activation-r1", "rollback-r1", "final-r1"], required=True
    )
    args = parser.parse_args()
    root = WORK / "evidence" / ("sdk-d49-" + args.phase)
    root.mkdir(exist_ok=False)
    old = args.phase == "rollback-r1"
    expected = OLD_SHA if old else NEW_SHA
    assert sha(BASE / "current/rocm" / SDK) == expected
    record = dict(
        status="running",
        phases=[],
        sources={p.name: sha(p) for p in TOOLS.glob("*.py")},
    )
    state = None
    guard = None
    with reservation("d49-" + args.phase):
        guarddir = root / "guard"
        with (root / "guard.log").open("w") as log:
            guard = subprocess.Popen(
                [
                    "python3",
                    str(TOOLS / "deadline_guard_d42.py"),
                    "--release",
                    os.uname().release,
                    "--phase",
                    "sdk-d49-" + args.phase,
                    "--timeout",
                    "600",
                    "--directory",
                    str(guarddir),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            for _ in range(100):
                if (guarddir / "state.json").exists():
                    break
                time.sleep(0.05)
            state = json.loads((guarddir / "state.json").read_text())
            assert state["status"] == "armed"
            start_all = time.monotonic()
            for profile in ("daily", "rccl-single"):
                cases = [("default", None, None, "gpu")]
                if not old:
                    cases += [("hip-negative", "HIP_VISIBLE_DEVICES", "-1", "cpu")]
                    if args.phase == "activation-r1":
                        cases += [
                            ("cuda-negative", "CUDA_VISIBLE_DEVICES", "-1", "cpu"),
                            ("rocr-negative", "ROCR_VISIBLE_DEVICES", "-1", "cpu"),
                            ("hip-zero", "HIP_VISIBLE_DEVICES", "0", "gpu"),
                            ("hip-terminated", "HIP_VISIBLE_DEVICES", "0,-1", "gpu"),
                        ]
                control = None
                for label, variable, value, kind in cases:
                    assert time.monotonic() - start_all < 420
                    folder = root / (profile + "-" + label)
                    folder.mkdir()
                    os.chown(folder, 1000, 1000)
                    before = subprocess.check_output(["dmesg"])
                    (folder / "kernel-before.log").write_bytes(before)
                    command = [
                        "/usr/local/bin/bc250-torch",
                        "--runtime-profile",
                        profile,
                        "--shm-size",
                        "256m",
                    ]
                    if variable:
                        command += ["--env", variable + "=" + value]
                    command += [
                        str(TOOLS / "sdk_d49_probe.py"),
                        "--kind",
                        kind,
                        "--sdk-sha",
                        expected,
                        "--output",
                        str(folder / "worker.json"),
                    ]
                    phase = dict(
                        name=folder.name, command=command, samples=[], containers=[]
                    )
                    record["phases"].append(phase)
                    start = time.monotonic()
                    with (folder / "worker.log").open("w") as log:
                        child = subprocess.Popen(
                            command,
                            env=dict(
                                os.environ,
                                PATH=str(TOOLS / "test-tools")
                                + ":"
                                + os.environ["PATH"],
                            ),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        phase["child_pid"] = child.pid
                        try:
                            while child.poll() is None:
                                phase["containers"] = sorted(
                                    set(phase["containers"]) | set(owned())
                                )
                                sample = resources()
                                phase["samples"].append(sample)
                                assert (
                                    sample["available_bytes"] >= 3 * 1024**3
                                    and sample["disk_free_bytes"] >= 5 * 1024**3
                                )
                                assert time.monotonic() - start < 60
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
                                os.killpg(child.pid, signal.SIGTERM)
                            child.wait(timeout=15)
                    phase["seconds"] = time.monotonic() - start
                    after = subprocess.check_output(["dmesg"])
                    (folder / "kernel-after.log").write_bytes(after)
                    phase["kernel_unchanged"] = before == after
                    assert phase["exit"] == 0 and phase["kernel_unchanged"], folder.name
                    worker = json.loads((folder / "worker.json").read_text())
                    assert (
                        worker["status"] == "passed"
                        and worker["sdk_sha256"] == expected
                    )
                    if kind == "gpu":
                        if control is None:
                            control = worker
                        else:
                            assert worker["digests"] == control["digests"]
                        d48 = json.loads(
                            (
                                WORK
                                / "evidence/sdk-d48-tests-r3"
                                / (profile + "-control")
                                / "worker.json"
                            ).read_text()
                        )
                        assert worker["digests"] == d48["digests"]
                    phase["worker"] = worker
                    preflight("d49-after-" + folder.name)
            if old:
                env = dict(
                    os.environ,
                    HIP_VISIBLE_DEVICES="-1",
                    HOME="/home/nik",
                    USER="nik",
                    LOGNAME="nik",
                )

                def as_user():
                    os.setgroups([])
                    os.setgid(1000)
                    os.setuid(1000)

                result = subprocess.run(
                    [
                        "/usr/local/bin/bc250-torch",
                        "-c",
                        "raise RuntimeError('WORKER_MUST_NOT_RUN')",
                    ],
                    env=env,
                    preexec_fn=as_user,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                assert (
                    result.returncode == 2
                    and "recorded ROCr import crash" in result.stderr
                    and "WORKER_MUST_NOT_RUN" not in result.stderr
                )
                record["legacy_refusal"] = dict(
                    exit=result.returncode, stderr=result.stderr
                )
            record.update(status="passed", seconds=time.monotonic() - start_all)
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            assert not owned(), "Owned workload remains; recovery stays armed"
            try:
                record["postflight"] = preflight("d49-final")
            except Exception as error:
                record.update(status="failed", postflight_error=repr(error))
            assert state is not None
            write(
                guarddir / "ack.json",
                {key: state[key] for key in ("boot_id", "nonce", "phase")},
            )
            assert guard.wait(timeout=10) == 0
            record["guard"] = json.loads((guarddir / "state.json").read_text())
            write(root / "result.json", record)
    assert record["status"] == "passed"


if __name__ == "__main__":
    main()
