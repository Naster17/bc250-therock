#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded CPU-only dependency replay, retained Gloo and version-input audit."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

WORK = Path("/home/nik/bc250work")
TOOLS = WORK / "tools/source-d50"
ROOT = WORK / "evidence/source-d50-r1"
GIB = 1024**3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def limits():
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    resource.setrlimit(resource.RLIMIT_AS, (2 * GIB, 2 * GIB))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(exist_ok=False)
    os.chown(ROOT, 1000, 1000)
    start = time.monotonic()
    admission = resources()
    assert (
        admission["available_bytes"] > 8 * GIB
        and admission["disk_free_bytes"] > 9 * GIB
    )
    package = TOOLS / "dependencies-r1"
    command = [
        "python3",
        str(package / "portable_dependencies_d50.py"),
        "replay",
        "--package",
        str(package),
        "--destination",
        str(ROOT / "replay"),
        "--output",
        str(ROOT / "replay.json"),
    ]
    record = dict(
        status="running",
        command=command,
        samples=[],
        admission=admission,
        driver_sha256=sha(package / "portable_dependencies_d50.py"),
        manifest_sha256=sha(package / "manifest.json"),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    try:
        with (ROOT / "worker.log").open("w") as log:
            child = subprocess.Popen(
                command,
                preexec_fn=limits,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record["child_pid"] = child.pid
            try:
                while child.poll() is None:
                    sample = resources()
                    record["samples"].append(sample)
                    assert (
                        sample["available_bytes"] >= 3 * GIB
                        and sample["disk_free_bytes"] >= 5 * GIB
                    )
                    assert (
                        time.monotonic() - start < 420
                    ), "CPU source replay wall deadline"
                    time.sleep(0.1)
                assert child.returncode == 0, child.returncode
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
        replay = json.loads((ROOT / "replay.json").read_text())
        assert replay["status"] == "passed"
        record["replay"] = replay
        base = Path("/opt/bc250-stack/current").resolve()
        with (base / ".runtime.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            assert base == Path("/opt/bc250-stack/candidates/sdk-d49-r1/profiles/daily")
            records = {}
            for profile, tree in [
                ("daily", "pytorch-quant-b052"),
                ("rccl-single", "pytorch-quant-b053"),
            ]:
                source = WORK / "rocm-10.2-gfx1013" / tree
                gloo = (
                    WORK
                    / "rocm-10.2-gfx1013/pytorch-work/third_party/gloo/gloo/cuda.cu"
                )
                # This is the retained pre-HIPify dependency source, not a newly
                # generated code file or a native build invocation.
                assert sha(gloo) == replay["gloo_cuda_sha256"]
                cmake = source / "build/CMakeCache.txt"
                values = {
                    line.split(":", 1)[0]: line.partition("=")[2]
                    for line in cmake.read_text().splitlines()
                    if ":" in line and "=" in line and not line.startswith(("#", "//"))
                }
                assert (
                    values["USE_FLASH_ATTENTION"] == "0"
                    and values["USE_MEM_EFF_ATTENTION"] == "0"
                )
                assert values["USE_GLOO"] == "1" and values["USE_TENSORPIPE"] == "1"
                version = source / "torch/version.py"
                records[profile] = dict(
                    gloo_source=str(gloo),
                    gloo_cuda_sha256=sha(gloo),
                    cmake_sha256=sha(cmake),
                    flags={
                        k: values[k]
                        for k in [
                            "USE_FLASH_ATTENTION",
                            "USE_MEM_EFF_ATTENTION",
                            "USE_GLOO",
                            "USE_TENSORPIPE",
                            "USE_ROCM",
                            "USE_DISTRIBUTED",
                            "CMAKE_BUILD_TYPE",
                        ]
                    },
                    version_sha256=sha(version),
                    version_text=version.read_text(),
                )
            record["retained_builds"] = records
            record["sdk_sha256"] = sha(base / "rocm/lib/librocprofiler-sdk.so.1.4.2")
            assert (
                record["sdk_sha256"]
                == "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b"
            )
        record.update(status="passed", seconds=time.monotonic() - start)
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        (ROOT / "result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
