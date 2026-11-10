#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded read-only external input export through an owned FIFO, no disk duplicate."""
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

WORK = Path("/home/nik/bc250work")
ROOT = WORK / "evidence/source-d53-r1"
TOOLS = WORK / "tools/source-d53"


def sample():
    available = next(
        x.split()[1]
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemAvailable:")
    )
    return dict(
        available_bytes=int(available) * 1024,
        disk_free_bytes=shutil.disk_usage(WORK).free,
    )


def limits():
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    os.sched_setaffinity(0, {0, 1})
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(exist_ok=False)
    os.chown(ROOT, 1000, 1000)
    fifo = ROOT / "payload.fifo"
    os.mkfifo(fifo, 0o600)
    os.chown(fifo, 1000, 1000)
    roots = dict(
        rocm=str(WORK / "rocm-10.2-gfx1013/build/dist/rocm"),
        rccl=str(WORK / "rocm-10.2-gfx1013/rccl-b034/install-r1"),
    )
    # Exact retained toolchain/ccache helpers used by the recorded build mounts.
    roots["torch_toolchain.cmake"] = str(
        WORK / "rocm-10.2-gfx1013/helpers/torch_toolchain.cmake"
    )
    for name in ("setup_ccache.py", "posix_ccache_compiler_check.py"):
        roots[name] = str(WORK / "rocm-10.2-gfx1013/source/build_tools" / name)
    (ROOT / "roots.json").write_text(json.dumps(roots, indent=2) + "\n")
    record = dict(
        status="running",
        pid=os.getpid(),
        samples=[],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        active_prefix=os.path.realpath("/opt/bc250-stack/current"),
    )
    started = time.monotonic()
    with (WORK / "evidence/.board-reservation.lock").open("a") as board, Path(
        "/opt/bc250-stack/current/.runtime.lock"
    ).open("r") as runtime:
        fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runtime, fcntl.LOCK_SH | fcntl.LOCK_NB)
        command = [
            "python3",
            str(TOOLS / "external_kit_d53.py"),
            "create",
            "--roots",
            str(ROOT / "roots.json"),
            "--output",
            str(ROOT / "manifest.json"),
        ]
        record["command"] = command
        child = None
        try:
            point = sample()
            assert (
                point["available_bytes"] >= 4 * 1024**3
                and point["disk_free_bytes"] >= 5 * 1024**3
            )
            # Shell FIFO open happens in the child, so a missing reader is bounded too.
            with (ROOT / "worker.log").open("w") as log:
                child = subprocess.Popen(
                    ["sh", "-c", 'exec "$@" > "$D53_FIFO"', "sh", *command],
                    env=dict(os.environ, D53_FIFO=str(fifo), HOME="/home/nik"),
                    preexec_fn=limits,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                )
                record["child_pid"] = child.pid
                (ROOT / "status.json").write_text(json.dumps(record, indent=2) + "\n")
                while child.poll() is None:
                    point = sample()
                    record["samples"].append(point)
                    assert (
                        point["available_bytes"] >= 3 * 1024**3
                        and point["disk_free_bytes"] >= 5 * 1024**3
                    )
                    assert time.monotonic() - started < 900
                    time.sleep(0.1)
                assert child.returncode == 0, child.returncode
            record.update(
                status="passed",
                seconds=time.monotonic() - started,
                active_prefix_after=os.path.realpath("/opt/bc250-stack/current"),
                boot_after=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            )
            assert (
                record["active_prefix_after"] == record["active_prefix"]
                and record["boot_after"] == record["boot_id"]
            )
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            (ROOT / "result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
