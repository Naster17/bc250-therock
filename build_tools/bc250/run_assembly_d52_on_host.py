#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded fresh source materialization; no GPU, code generation or native build."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

WORK = Path("/home/nik/bc250work")
TOOLS = WORK / "tools/source-d52"
ROOT = WORK / "evidence/source-d52-assembly-r1"


def sample():
    return dict(
        available_bytes=int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024,
        disk_free_bytes=shutil.disk_usage(WORK).free,
    )


def limits():
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        choices=["source-d52-assembly-r1", "source-d52-assembly-r2"],
        default="source-d52-assembly-r1",
    )
    args = parser.parse_args()
    ROOT = WORK / "evidence" / args.label
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(exist_ok=False)
    os.chown(ROOT, 1000, 1000)
    point = sample()
    assert (
        point["available_bytes"] > 8 * 1024**3
        and point["disk_free_bytes"] > 10 * 1024**3
    )
    command = [
        "python3",
        str(TOOLS / "assemble_source_d52.py"),
        "--parent",
        str(WORK / "tools/source-d47/portable-torch-r1"),
        "--dependencies",
        str(WORK / "tools/source-d50/dependencies-r1"),
        "--inputs",
        str(WORK / "tools/source-d51/build-inputs-r1"),
        "--destination",
        str(ROOT / "assembled"),
        "--output",
        str(ROOT / "assembly.json"),
    ]
    record = dict(status="running", pid=os.getpid(), command=command, samples=[])
    started = time.monotonic()
    try:
        with (ROOT / "worker.log").open("w") as log:
            child = subprocess.Popen(
                command,
                env=dict(
                    os.environ,
                    HOME="/home/nik",
                    XDG_CONFIG_HOME=str(ROOT / "isolated-config"),
                ),
                preexec_fn=limits,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record["child_pid"] = child.pid
            try:
                while child.poll() is None:
                    point = sample()
                    record["samples"].append(point)
                    assert (
                        point["available_bytes"] >= 3 * 1024**3
                        and point["disk_free_bytes"] >= 5 * 1024**3
                    )
                    assert time.monotonic() - started < 480
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
        result = json.loads((ROOT / "assembly.json").read_text())
        assert result["status"] == "passed"
        record.update(
            status="passed", assembly=result, seconds=time.monotonic() - started
        )
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        (ROOT / "result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
