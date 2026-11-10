#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded D43 phase guardian; preserves user processes and installed profiles."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from docker_preflight import docker_containers
from board_preflight import reservation
from capacity_plan_d41 import CONFIG, CONTROL, parameters
from capacity_plan_d42 import (
    admit,
    headroom,
    CHILD_SECONDS,
    GUARD_SECONDS,
    HOST_REGISTRY_SHA256,
)
import shutil


def preflight():
    assert not docker_containers(), "Unrelated Docker containers are running"
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    available = (
        int(
            next(
                v.split()[1]
                for v in Path("/proc/meminfo").read_text().splitlines()
                if v.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    assert available > 8 * 1024**3, available
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            links = []
            for fd in (proc / "fd").iterdir():
                try:
                    links.append(os.readlink(fd))
                except OSError:
                    pass
            if any("/dev/kfd" in x or "/dev/dri/render" in x for x in links):
                assert proc.stat().st_uid == 0 and (
                    proc / "comm"
                ).read_text().startswith("cyan-skillfish"), proc.name
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", choices=["control", "large"], required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    a.transport = "none"
    config = CONTROL if a.shape == "control" else CONFIG
    count = parameters(config)
    from qualification_d43 import proof

    item = proof(a.shape, 0 if os.uname().nodename == "bc250" else 1)
    a.mode = "fsdp"
    a.phase = "measure"
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    assert a.label.replace("-", "").isalnum()
    preflight()
    rank = 0 if os.uname().nodename == "bc250" else 1
    series = Path("/home/nik/bc250work/evidence") / "distributed-d43-r1"
    root = series / a.label
    root.mkdir(parents=True)
    os.chown(root, 1000, 1000)
    lock = Path(
        "/opt/bc250-stack/current/.runtime.lock"
        if rank == 0
        else "/opt/bc250-stack/migration-control/.runtime.lock"
    )
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    registry_before = registry.read_bytes()
    docker_before = docker_containers()
    assert not docker_before
    before = subprocess.check_output(["dmesg"])
    (root / "kernel-before.log").write_bytes(before)
    state = None
    guard = None
    child = None
    result = dict(
        status="running",
        rank=rank,
        mode=a.mode,
        shape=a.shape,
        transport=a.transport,
        phase=a.phase,
        kernel=os.uname().release,
        guardian_pid=os.getpid(),
    )
    tools = Path("/usr/local/bin/bc250-distributed").resolve().parent
    assert tools.name == "distributed-d09"
    result["selected_tools"] = str(tools)
    result["launcher_sha256"] = hashlib.sha256(
        (tools / "bc250_distributed.py").read_bytes()
    ).hexdigest()
    consumers = Path("/opt/bc250-stack/current/python-consumers")
    label = "d43-r1-" + a.label

    def owned_containers():
        return subprocess.check_output(
            ["podman", "ps", "-q", "--filter", "label=bc250.d43=" + label], text=True
        ).split()

    command = [
        "python3",
        "/home/nik/bc250work/tools/distributed-d43/distributed_d43_bounded_launch.py",
        "--node-rank",
        str(rank),
        "--master-port",
        str(a.port),
        "--env-file",
        str(consumers / "training.env"),
        "--mount-ro",
        str(consumers),
        "--env",
        "HF_HUB_OFFLINE=1",
        "--shm-size",
        "256m",
        "/home/nik/bc250work/tools/distributed-d43/distributed_d43_train.py",
        "--shape",
        a.shape,
        "--output",
        str(root / "rank.json"),
    ]
    result["command"] = command
    result["registry_sha256"] = hashlib.sha256(registry_before).hexdigest()
    with reservation("d43-" + a.label), lock.open("r") as fd:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert registry.read_bytes() == registry_before
        available = (
            lambda: int(
                next(
                    v.split()[1]
                    for v in Path("/proc/meminfo").read_text().splitlines()
                    if v.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        result["admission"] = admit(a.shape, available(), shutil.disk_usage(root).free)
        assert HOST_REGISTRY_SHA256[rank] == result["registry_sha256"]
        result["D43_qualification_sha256"] = item["sha256"]
        try:
            guarddir = root / "guard"
            guard = subprocess.Popen(
                [
                    "python3",
                    "/home/nik/bc250work/tools/distributed-d43/deadline_guard_d42.py",
                    "--release",
                    os.uname().release,
                    "--phase",
                    "distributed-d43-" + a.label,
                    "--timeout",
                    str(GUARD_SECONDS),
                    "--directory",
                    str(guarddir),
                ],
                stdout=(root / "guard.log").open("w"),
                stderr=subprocess.STDOUT,
            )
            for _ in range(100):
                if (guarddir / "state.json").exists():
                    break
                time.sleep(0.05)
            state = json.loads((guarddir / "state.json").read_text())
            assert state["status"] == "armed"
            result["boot_id"] = state["boot_id"]
            phase_start = time.monotonic()
            with (root / "gpu.log").open("w") as log:
                child_env = dict(os.environ)
                child_env["BC250_D43_LABEL"] = label
                child_env["BC250_D05_CANDIDATE_TOOL"] = str(tools)
                child = subprocess.Popen(
                    command,
                    cwd=Path("/home/nik/bc250work/tools/distributed-d43"),
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                for _ in range(200):
                    ids = owned_containers()
                    if ids:
                        assert len(ids) == 1
                        result["container_id"] = ids[0]
                        break
                    if child.poll() is not None:
                        break
                    time.sleep(0.1)
                assert result.get("container_id"), "GPU container did not start"
                pid = subprocess.check_output(
                    [
                        "podman",
                        "inspect",
                        "--format",
                        "{{.State.Pid}}",
                        result["container_id"],
                    ],
                    text=True,
                ).strip()
                cg = Path("/sys/fs/cgroup") / Path(
                    "/proc", pid, "cgroup"
                ).read_text().strip().split("::", 1)[1].lstrip("/")
                assert int((cg / "memory.max").read_text()) == 10 * 1024**3
                result["container_cgroup"] = str(cg)
                deadline = time.monotonic() + CHILD_SECONDS
                readings = []
                try:
                    while child.poll() is None:
                        if time.monotonic() > deadline:
                            subprocess.run(
                                [
                                    "podman",
                                    "stop",
                                    "--time",
                                    "5",
                                    result["container_id"],
                                ],
                                check=True,
                                timeout=15,
                            )
                            child.wait(timeout=15)
                            raise TimeoutError("D43 audit deadline")
                        try:
                            current = dict(
                                memory_bytes=int((cg / "memory.current").read_text()),
                                available_bytes=available(),
                                free_bytes=shutil.disk_usage(root).free,
                            )
                        except FileNotFoundError:
                            time.sleep(0.1)
                            continue
                        readings.append(current)
                        try:
                            headroom(
                                current["memory_bytes"],
                                current["available_bytes"],
                                current["free_bytes"],
                            )
                        except MemoryError:
                            result["headroom_abort"] = current
                            assert owned_containers() == [result["container_id"]]
                            subprocess.run(
                                [
                                    "podman",
                                    "stop",
                                    "--time",
                                    "5",
                                    result["container_id"],
                                ],
                                check=True,
                                timeout=15,
                            )
                            child.wait(timeout=15)
                            raise
                        if time.monotonic() > deadline:
                            subprocess.run(
                                [
                                    "podman",
                                    "stop",
                                    "--time",
                                    "5",
                                    result["container_id"],
                                ],
                                check=True,
                                timeout=15,
                            )
                            child.wait(timeout=15)
                            raise TimeoutError("D43 audit deadline")
                        time.sleep(0.1)
                    code = child.returncode
                finally:
                    result["headroom_samples"] = readings
            result["guarded_phase_seconds"] = time.monotonic() - phase_start
            assert code == 0, code
            r = json.loads((root / "rank.json").read_text())
            assert r["status"] == "passed"
            result.update(status="passed", worker=r)
            text = (root / "gpu.log").read_text()
            result["agent_shutdown_clean"] = not any(
                x in text for x in ["DistNetworkError", "Error waiting on exit barrier"]
            )
            assert result["agent_shutdown_clean"], "Agent shutdown failed"
        except BaseException as e:
            result.update(status="failed", error=repr(e))
            raise
        finally:
            # Acknowledge recovery only once the owned container actually exits.
            for _ in range(100):
                if not owned_containers():
                    break
                time.sleep(0.1)
            assert (
                not owned_containers()
            ), "Owned container remains; recovery guard stays armed"
            # A new unrelated container invalidates exclusivity but is never stopped here.
            foreign = subprocess.check_output(["podman", "ps", "-q"], text=True).split()
            if foreign:
                result.update(status="failed", foreign_containers=foreign)
            try:
                docker_after = docker_containers()
                result["docker_containers"] = dict(
                    before=docker_before, after=docker_after
                )
                if docker_after:
                    result["status"] = "failed"
            except Exception as exc:
                result.update(status="failed", docker_error=repr(exc))
            after = subprocess.check_output(["dmesg"])
            (root / "kernel-after.log").write_bytes(after)
            result["kernel_log_unchanged"] = before == after
            result["registry_unchanged"] = registry_before == registry.read_bytes()
            if not result["kernel_log_unchanged"] or not result["registry_unchanged"]:
                result["status"] = "failed"
            if state is not None:
                (root / "guard/ack.json").write_text(
                    json.dumps({k: state[k] for k in ["nonce", "boot_id", "phase"]})
                )
                guard.wait(timeout=10)
            (root / "final.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
