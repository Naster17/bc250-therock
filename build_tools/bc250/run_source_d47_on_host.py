#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded CPU-only portable replay and read-only SDK candidate admission."""
import fcntl
import argparse
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
TOOLS = WORK / "tools/source-d47"
ROOT = WORK / "evidence/source-d47-r1"
GIB = 1024**3


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
    return dict(available_bytes=memory, disk_free_bytes=shutil.disk_usage(WORK).free)


def limits():
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    resource.setrlimit(resource.RLIMIT_AS, (2 * GIB, 2 * GIB))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="source-d47-r1")
    args = parser.parse_args()
    assert (
        args.label.startswith("source-d47-") and args.label.replace("-", "").isalnum()
    )
    ROOT = WORK / "evidence" / args.label
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(exist_ok=False)
    os.chown(ROOT, 1000, 1000)
    admission = resources()
    assert (
        admission["available_bytes"] > 8 * GIB
        and admission["disk_free_bytes"] > 7 * GIB
    )
    command = [
        "python3",
        str(TOOLS / "portable_torch_d47.py"),
        "replay",
        "--package",
        str(TOOLS / "portable-torch-r1"),
        "--destination",
        str(ROOT / "replay"),
        "--output",
        str(ROOT / "replay.json"),
    ]
    result = dict(
        status="running",
        admission=admission,
        command=command,
        samples=[],
        source_sha256=hashlib.sha256(
            (TOOLS / "portable_torch_d47.py").read_bytes()
        ).hexdigest(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    start = time.monotonic()
    try:
        with (ROOT / "worker.log").open("w") as log:
            child = subprocess.Popen(
                command,
                preexec_fn=limits,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            result["child_pid"] = child.pid
            try:
                while child.poll() is None:
                    sample = resources()
                    result["samples"].append(sample)
                    assert (
                        sample["available_bytes"] >= 3 * GIB
                        and sample["disk_free_bytes"] >= 5 * GIB
                    ), sample
                    assert time.monotonic() - start < 180, "Source replay wall deadline"
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
        result["replay"] = json.loads((ROOT / "replay.json").read_text())
        assert result["replay"]["status"] == "passed"
        daily = Path("/opt/bc250-stack/current").resolve()
        with (daily / ".runtime.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            build = WORK / "rocm-10.2-gfx1013/build/profiler/rocprofiler-sdk/build"
            config = (build / "CMakeCache.txt").read_text().splitlines()
            sdk = daily / "rocm/lib/librocprofiler-sdk.so"
            current = resources()
            planned_disk, planned_memory = 4 * GIB, 4 * GIB
            source_manifest = json.loads((daily / "runtime.json").read_text())
            budget = dict(
                status=(
                    "resource-admitted-recipe-audit-pending"
                    if current["disk_free_bytes"] >= 5 * GIB + planned_disk
                    and current["available_bytes"] >= 3 * GIB + planned_memory
                    else "refused"
                ),
                resources=current,
                planned_peak_extra_disk_bytes=planned_disk,
                planned_candidate_memory_bytes=planned_memory,
                host_reserve_bytes=3 * GIB,
                disk_reserve_bytes=5 * GIB,
                jobs=1,
                link_jobs=1,
                ccache_maxsize="20G; preserve existing cache/statistics; owned stop if actual disk reserve is threatened",
                image=source_manifest["image"],
                build=str(build),
                sdk_bytes=sdk.stat().st_size,
                selected_cache=[
                    line
                    for line in config
                    if line.startswith(
                        (
                            "ROCPROFILER_BUILD_CI:",
                            "CMAKE_BUILD_TYPE:",
                            "CMAKE_CXX_COMPILER:",
                            "CMAKE_CXX_COMPILER_LAUNCHER:",
                            "CMAKE_LINKER:",
                        )
                    )
                ],
                original_build_and_source="read-only; isolated modified source/object/link output required",
                native_build_started=False,
                qualification="candidate negative imports + valid GPU/reference/profiler gates before any activation",
            )
            if (build / "compile_commands.json").exists():
                entries = json.loads((build / "compile_commands.json").read_text())
                budget["agent_compile_commands"] = [
                    e
                    for e in entries
                    if e["file"].endswith("source/lib/rocprofiler-sdk/agent.cpp")
                ]
            ninja = shutil.which("ninja")
            if ninja:
                target = "lib/librocprofiler-sdk.so.1.4.2"
                commands = subprocess.run(
                    [ninja, "-C", str(build), "-t", "commands", target],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                (ROOT / "sdk-target-commands.log").write_text(
                    commands.stdout + commands.stderr
                )
                budget["ninja_commands_exit"] = commands.returncode
            else:
                budget["ninja_commands_exit"] = (
                    "host ninja unavailable; derive in matching image"
                )
            (ROOT / "sdk-build-admission.json").write_text(
                json.dumps(budget, indent=2) + "\n"
            )
            result["sdk_build_admission"] = budget["status"]
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                seconds=result["seconds"],
                sdk_admission=result["sdk_build_admission"],
            )
        )
    )


if __name__ == "__main__":
    main()
