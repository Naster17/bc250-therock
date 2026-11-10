#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve the original build image with storage admission and bounded owned work."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from board_preflight import preflight
from verify_oci_d52 import IMAGE, verify

WORK = Path("/home/nik/bc250work")
ROOT = WORK / "evidence/source-d52-image-r1"


def sample():
    memory = (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return dict(available_bytes=memory, disk_free_bytes=shutil.disk_usage(WORK).free)


def affinity():
    os.sched_setaffinity(0, {0, 1})


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(exist_ok=False)
    (ROOT / "scratch").mkdir()
    record = dict(status="running", pid=os.getpid(), samples=[])
    with (WORK / "evidence/.board-reservation.lock").open("a") as board, Path(
        "/opt/bc250-stack/current/.runtime.lock"
    ).open("r") as runtime:
        fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runtime, fcntl.LOCK_SH | fcntl.LOCK_NB)
        record["preflight"] = preflight("d52-builder-export")
        image = json.loads(
            subprocess.check_output(["podman", "image", "inspect", IMAGE])
        )[0]
        assert image["Id"] == IMAGE
        point = sample()
        required = 2 * image["Size"] + 4 * 1024**3 + 5 * 1024**3
        assert point["disk_free_bytes"] > required, (point, required)
        record["admission"] = dict(
            image_size=image["Size"], required_free_bytes=required, resources=point
        )
        archive = ROOT / "native-builder-28ad36e6.oci.tar"
        command = [
            "podman",
            "save",
            "--format",
            "oci-archive",
            "--output",
            str(archive),
            IMAGE,
        ]
        record["command"] = command
        started = time.monotonic()
        try:
            with (ROOT / "save.log").open("w") as log:
                child = subprocess.Popen(
                    command,
                    preexec_fn=affinity,
                    start_new_session=True,
                    env=dict(os.environ, TMPDIR=str(ROOT / "scratch"), GOMAXPROCS="2"),
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
                        assert time.monotonic() - started < 420
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
            record["verification"] = verify(archive)
            assert (
                json.loads(
                    subprocess.check_output(["podman", "image", "inspect", IMAGE])
                )[0]["Id"]
                == IMAGE
            )
            record.update(
                status="passed",
                seconds=time.monotonic() - started,
                postflight=preflight("d52-builder-export-final"),
            )
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            (ROOT / "result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
