#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve inspected workspace copies before retiring them."""
import argparse
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
TOOLS = WORK / "tools/storage-cleanup-20261110"


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


def limits(staging):
    if not staging:
        os.setgroups([])
        os.setgid(1000)
        os.setuid(1000)
    os.sched_setaffinity(0, {0, 1})
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    cpu = 1200 if staging else 900
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", action="store_true")
    staging = parser.parse_args().staging
    label = "staging-r2" if staging else "archive-r1"
    root = WORK / "evidence" / f"storage-cleanup-20261110-{label}"
    root.mkdir(exist_ok=False)
    os.chown(root, 1000, 1000)
    fifo = root / "payload.fifo"
    os.mkfifo(fifo, 0o600)
    os.chown(fifo, 1000, 1000)
    roots = {
        "d47-r1": str(WORK / "evidence/source-d47-r1/replay"),
        "d47-r2": str(WORK / "evidence/source-d47-r2/replay"),
        "d50-r1": str(WORK / "evidence/source-d50-r1/replay"),
        "d52-r1": str(WORK / "evidence/source-d52-assembly-r1/assembled"),
        "d52-r2": str(WORK / "evidence/source-d52-assembly-r2/assembled"),
    }
    if staging:
        roots = json.loads((TOOLS / "staging-roots.json").read_text())
        allowed = (
            WORK / "rocm-10.2-gfx1013/tmp",
            WORK / "evidence",
            WORK / "tools",
        )
        assert roots and all(Path(path).parent in allowed for path in roots.values())
    (root / "roots.json").write_text(json.dumps(roots, indent=2) + "\n")
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
            *(["--deduplicate"] if staging else []),
            "--roots",
            str(root / "roots.json"),
            "--output",
            str(root / "manifest.json"),
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
            with (root / "worker.log").open("w") as log:
                child = subprocess.Popen(
                    ["sh", "-c", 'exec "$@" > "$D53_FIFO"', "sh", *command],
                    env=dict(os.environ, D53_FIFO=str(fifo), HOME="/home/nik"),
                    preexec_fn=lambda: limits(staging),
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                )
                record["child_pid"] = child.pid
                (root / "status.json").write_text(json.dumps(record, indent=2) + "\n")
                while child.poll() is None:
                    point = sample()
                    record["samples"].append(point)
                    assert (
                        point["available_bytes"] >= 3 * 1024**3
                        and point["disk_free_bytes"] >= 5 * 1024**3
                    )
                    assert time.monotonic() - started < (1800 if staging else 1200)
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
            (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
