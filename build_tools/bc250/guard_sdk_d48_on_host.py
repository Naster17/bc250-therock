#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded SDK control/candidate processes under independent GPU recovery."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from board_preflight import reservation, preflight

WORK = Path("/home/nik/bc250work")
TOOLS = WORK / "tools/sdk-d48"
ROOT = WORK / "evidence/sdk-d48-tests-r3"
BUILD = WORK / "evidence/sdk-d48-build-r2"
GIB = 1024**3


def owned():
    return subprocess.check_output(
        ["podman", "ps", "-q", "--filter", "label=bc250.d48=test-r3"],
        text=True,
        timeout=10,
    ).split()


def resources():
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return dict(available_bytes=memory, disk_free_bytes=shutil.disk_usage(ROOT).free)


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    assert not any(
        k in os.environ
        for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    )
    built = json.loads((BUILD / "build.json").read_text())
    assert built["status"] == "built-unqualified"
    prefix = BUILD / "prefix"
    if prefix.exists():
        assert (
            hashlib.sha256(
                (prefix / "lib/librocprofiler-sdk.so.1.4.2").read_bytes()
            ).hexdigest()
            == built["library_sha256"]
        )
    else:
        (prefix / "lib").mkdir(parents=True, exist_ok=False)
        os.link(
            BUILD / "librocprofiler-sdk.so.1.4.2",
            prefix / "lib/librocprofiler-sdk.so.1.4.2",
        )
        for name in ("rocm_sysdeps", "llvm"):
            (prefix / "lib" / name).symlink_to("/workspace/build/dist/rocm/lib/" + name)
        (prefix / "share").symlink_to("/workspace/build/dist/rocm/share")
    for name in ("librocprofiler-sdk.so", "librocprofiler-sdk.so.1"):
        path = prefix / "lib" / name
        if not path.exists():
            path.symlink_to("librocprofiler-sdk.so.1.4.2")
    ROOT.mkdir(exist_ok=False)
    before_sdk = hashlib.sha256(
        Path("/opt/bc250-stack/current/rocm/lib/librocprofiler-sdk.so").read_bytes()
    ).hexdigest()
    assert (
        before_sdk == "a6dc1c884c4f446e23a2afc8185b26ef9fd55b57cdd1d15324d46054b902b9c2"
    )
    record = dict(
        status="running",
        phases=[],
        candidate_sha256=built["library_sha256"],
        active_sdk_sha256=before_sdk,
        sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in TOOLS.glob("*.py")
        },
    )
    guard = None
    state = None
    with reservation("d48-sdk-tests"):
        guarddir = ROOT / "guard"
        with (ROOT / "guard.log").open("w") as log:
            guard = subprocess.Popen(
                [
                    "python3",
                    str(TOOLS / "deadline_guard_d42.py"),
                    "--release",
                    os.uname().release,
                    "--phase",
                    "sdk-d48",
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
            record["boot_id"] = state["boot_id"]
            started = time.monotonic()
            for profile in ("daily", "rccl-single"):
                cases = [
                    ("control", True, None, None, "gpu"),
                    ("default", False, None, None, "gpu"),
                ]
                cases += [
                    (
                        alias.lower() + "-negative",
                        False,
                        alias + "_VISIBLE_DEVICES",
                        "-1",
                        "cpu",
                    )
                    for alias in ("HIP", "CUDA", "ROCR")
                ]
                cases += [
                    (
                        alias.lower() + "-zero",
                        False,
                        alias + "_VISIBLE_DEVICES",
                        "0",
                        "gpu",
                    )
                    for alias in ("HIP", "CUDA", "ROCR")
                ]
                cases += [
                    (
                        alias.lower() + "-terminated",
                        False,
                        alias + "_VISIBLE_DEVICES",
                        "0,-1",
                        "gpu",
                    )
                    for alias in ("HIP", "CUDA", "ROCR")
                ]
                cases += [
                    (
                        alias.lower() + "-uuid",
                        False,
                        alias + "_VISIBLE_DEVICES",
                        "UUID",
                        "gpu",
                    )
                    for alias in ("HIP", "CUDA", "ROCR")
                ]
                if profile == "daily":
                    cases.insert(0, ("preload-diagnostic", True, None, None, "gpu"))
                control = None
                for label, baseline, variable, value, kind in cases:
                    assert time.monotonic() - started < 420, "Total test budget"
                    if value == "UUID":
                        if control["uuid"] == "GPU-XX":
                            record["uuid_selection"] = (
                                "unavailable: HSA UUID unsupported (GPU-XX); no synthetic identity substituted"
                            )
                            continue
                        value = control["uuid"]
                    folder = ROOT / (profile + "-" + label)
                    folder.mkdir()
                    os.chown(folder, 1000, 1000)
                    before = subprocess.check_output(["dmesg"])
                    (folder / "kernel-before.log").write_bytes(before)
                    command = [
                        "python3",
                        str(TOOLS / "sdk_d48_launch.py"),
                        "--runtime-profile",
                        profile,
                        "--shm-size",
                        "256m",
                    ]
                    if variable:
                        command += ["--env", variable + "=" + value]
                    command += [
                        str(TOOLS / "sdk_d48_probe.py"),
                        "--kind",
                        kind,
                        "--output",
                        str(folder / "worker.json"),
                    ]
                    phase = dict(
                        name=folder.name,
                        control=baseline,
                        kind=kind,
                        variable=variable,
                        value=value,
                        command=command,
                        samples=[],
                        containers=[],
                    )
                    record["phases"].append(phase)
                    if label == "preload-diagnostic":
                        command += ["--expect-missing-trace"]
                    env = dict(os.environ, BC250_D48_CONTROL=str(int(baseline)))
                    env["BC250_D48_PRELOAD_CONTROL"] = str(
                        int(label == "preload-diagnostic")
                    )
                    start = time.monotonic()
                    with (folder / "worker.log").open("w") as log:
                        child = subprocess.Popen(
                            command,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        phase["child_pid"] = child.pid
                        try:
                            while child.poll() is None:
                                ids = owned()
                                phase["containers"] = sorted(
                                    set(phase["containers"]) | set(ids)
                                )
                                sample = resources()
                                phase["samples"].append(sample)
                                assert (
                                    sample["available_bytes"] >= 3 * GIB
                                    and sample["disk_free_bytes"] >= 5 * GIB
                                ), sample
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
                    (folder / "kernel-after.log").write_bytes(after)
                    phase["kernel_unchanged"] = before == after
                    assert phase["exit"] == 0 and phase["kernel_unchanged"], folder.name
                    worker = json.loads((folder / "worker.json").read_text())
                    assert worker["status"] == "passed"
                    if baseline:
                        control = worker
                    elif kind == "gpu":
                        assert (
                            worker["digests"] == control["digests"]
                            and worker["uuid"] == control["uuid"]
                        )
                    phase["worker"] = worker
                    preflight("d48-after-" + folder.name)
            assert (
                hashlib.sha256(
                    Path(
                        "/opt/bc250-stack/current/rocm/lib/librocprofiler-sdk.so"
                    ).read_bytes()
                ).hexdigest()
                == before_sdk
            )
            record.update(status="passed", seconds=time.monotonic() - started)
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            assert not owned(), "Owned workload remains; recovery stays armed"
            try:
                record["postflight"] = preflight("d48-final")
            except Exception as error:
                record.update(status="failed", postflight_error=repr(error))
            assert state is not None
            (guarddir / "ack.json").write_text(
                json.dumps({key: state[key] for key in ("boot_id", "nonce", "phase")})
            )
            assert guard.wait(timeout=10) == 0
            record["guard"] = json.loads((guarddir / "state.json").read_text())
            (ROOT / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    assert record["status"] == "passed"


if __name__ == "__main__":
    main()
