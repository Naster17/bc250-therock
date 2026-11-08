#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cooperative board reservation and fail-closed workload checks; no workload stops."""
from contextlib import contextmanager
import fcntl
from functools import wraps
import json
import os
from pathlib import Path
import subprocess

from docker_preflight import docker_containers

WORK = Path("/home/nik/bc250work")


def preflight(attempt):
    assert os.geteuid() == 0
    assert not docker_containers(), "Unrelated default-local Docker workload"
    running = subprocess.check_output(
        ["podman", "ps", "-q"], text=True, timeout=10
    ).split()
    assert not running, ("Unrelated Podman workload", running)
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    available = (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    assert available > 8 * 1024**3, available
    clients = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            links = []
            for fd in (proc / "fd").iterdir():
                try:
                    links.append(str(fd.readlink()))
                except FileNotFoundError:
                    continue  # A closed fd is a normal process race.
            if any("/dev/kfd" in x or "/dev/dri/render" in x for x in links):
                clients.append(
                    dict(
                        pid=int(proc.name),
                        uid=proc.stat().st_uid,
                        comm=(proc / "comm").read_text().strip(),
                    )
                )
        except FileNotFoundError:
            continue  # A process that exited no longer owns the device.
    assert all(
        p["uid"] == 0 and p["comm"].startswith("cyan-skillfish") for p in clients
    ), clients
    return dict(
        attempt=attempt,
        memory_available_bytes=available,
        gpu_clients=clients,
        docker_containers=[],
        podman_containers=[],
        watchdog="active",
    )


@contextmanager
def reservation(attempt):
    directory = WORK / "evidence"
    directory.mkdir(exist_ok=True)
    lock = Path(
        "/opt/bc250-stack/current/.runtime.lock"
        if os.uname().nodename == "bc250"
        else "/opt/bc250-stack/migration-control/.runtime.lock"
    )
    assert os.uname().nodename in ["bc250", "bc500"]
    with (directory / ".board-reservation.lock").open("a") as owner, lock.open(
        "r"
    ) as runtime:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runtime, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield preflight(attempt)


def reserved(function):
    @wraps(function)
    def run(label, *args, **kwargs):
        with reservation(label):
            return function(label, *args, **kwargs)

    return run


if __name__ == "__main__":
    with reservation("plan01-idle-control") as result:
        print(json.dumps(result, sort_keys=True))
