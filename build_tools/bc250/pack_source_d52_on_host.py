#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create a bounded Gitless source snapshot from the passed private assembly."""
import json
import os
import signal
import subprocess
import time

from run_assembly_d52_on_host import WORK, TOOLS, limits, sample

ROOT = WORK / "evidence/source-d52-package-r1"
ASSEMBLY = WORK / "evidence/source-d52-assembly-r2/assembled"


def main():
    ROOT.mkdir(exist_ok=False)
    os.chown(ROOT, 1000, 1000)
    assert json.loads((ASSEMBLY / "assembly.json").read_text())["status"] == "passed"
    point = sample()
    assert (
        point["available_bytes"] > 8 * 1024**3
        and point["disk_free_bytes"] > 7 * 1024**3
    )
    command = [
        "python3",
        str(TOOLS / "source_kit_d52.py"),
        "create",
        "--archive",
        str(ROOT / "torch-source-d52-r1.tar.gz"),
        "--assembly",
        str(ASSEMBLY),
        "--output",
        str(ROOT / "package.json"),
    ]
    record = dict(status="running", pid=os.getpid(), command=command, samples=[])
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
            record["child_pid"] = child.pid
            try:
                while child.poll() is None:
                    point = sample()
                    record["samples"].append(point)
                    assert (
                        point["available_bytes"] >= 3 * 1024**3
                        and point["disk_free_bytes"] >= 5 * 1024**3
                    )
                    assert time.monotonic() - start < 480
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
        record.update(
            status="passed",
            package=json.loads((ROOT / "package.json").read_text()),
            seconds=time.monotonic() - start,
        )
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        (ROOT / "result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
