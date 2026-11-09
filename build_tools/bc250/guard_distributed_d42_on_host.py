#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded D42 phase guardian; preserves user processes and installed profiles."""
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
from artifact_d35 import load_manifest
from capacity_plan_d41 import CONFIG, CONTROL, parameters
from capacity_plan_d42 import (
    descriptor,
    admit,
    headroom,
    CHILD_SECONDS,
    GUARD_SECONDS,
    HOST_REGISTRY_SHA256,
)
from reference_transport_d40 import RemoteStore, fetch_manifest
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
    p.add_argument("--ready-port", type=int, required=True)
    a = p.parse_args()
    a.transport = "remote"
    config = CONTROL if a.shape == "control" else CONFIG
    count = parameters(config)
    proof = descriptor(a.shape)
    a.mode = "fsdp"
    a.phase = "qualify"
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    assert a.label.replace("-", "").isalnum()
    preflight()
    rank = 0 if os.uname().nodename == "bc250" else 1
    series = Path("/home/nik/bc250work/evidence") / "distributed-d42-r5"
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
    server = None
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
    label = "d42-r5-" + a.label

    def owned_containers():
        return subprocess.check_output(
            ["podman", "ps", "-q", "--filter", "label=bc250.d42=" + label], text=True
        ).split()

    command = [
        "python3",
        "/home/nik/bc250work/tools/distributed-d42/distributed_d42_bounded_launch.py",
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
        "/home/nik/bc250work/tools/distributed-d42/distributed_d42_capacity.py",
        "--shape",
        a.shape,
        "--ready-port",
        str(a.ready_port),
        "--transport",
        "local" if rank == 0 else "remote",
        "--mode",
        a.mode,
        "--phase",
        a.phase,
        "--output",
        str(root / "rank.json"),
    ]
    result["command"] = command
    result["registry_sha256"] = hashlib.sha256(registry_before).hexdigest()
    with reservation("d42-" + a.label), lock.open("r") as fd:
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
        reference_path = Path("/home/nik/bc250work/evidence") / (
            "distributed-d35-r1/cpu-reference/reference/manifest.json"
            if a.shape == "control"
            else "distributed-d41-r1/large-oracle/reference/manifest.json"
        )
        pin = proof["manifest_sha256"]
        if rank == 1:
            reference_path = root / "remote-reference/manifest.json"
            deadline = time.monotonic() + 180
            while True:
                try:
                    fetch_manifest(
                        "http://192.168.1.104:29842",
                        reference_path,
                        pin,
                        proof["manifest_bytes"],
                    )
                    break
                except ConnectionRefusedError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.2)
            remote = RemoteStore("http://192.168.1.104:29842", reference_path, pin)
            import artifact_d35 as artifact

            native = artifact.verify_blob
            artifact.verify_blob = remote.verify
            try:
                manifest = load_manifest(
                    reference_path, pin, config, count, check_sources=False
                )
            finally:
                artifact.verify_blob = native
            result["remote_preflight"] = remote.stats
            command += ["--endpoint", "http://192.168.1.104:29842"]
        else:
            manifest = load_manifest(
                reference_path, pin, config, count, check_sources=False
            )
        assert manifest == proof["manifest"]
        result["CPU_artifact_manifest_sha256"] = pin
        command += ["--reference", str(reference_path), "--reference-sha256", pin]
        try:
            guarddir = root / "guard"
            guard = subprocess.Popen(
                [
                    "python3",
                    "/home/nik/bc250work/tools/distributed-d42/deadline_guard_d42.py",
                    "--release",
                    os.uname().release,
                    "--phase",
                    "distributed-d42-" + a.label,
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
            if rank == 0 and a.transport == "remote":
                server = subprocess.Popen(
                    [
                        "python3",
                        "/home/nik/bc250work/tools/distributed-d42/serve_reference_d42.py",
                        "--shape",
                        a.shape,
                        "--reference",
                        str(reference_path),
                        "--sha256",
                        pin,
                        "--output",
                        str(root / "source.json"),
                    ],
                    user=1000,
                    group=1000,
                    stdin=subprocess.DEVNULL,
                    stdout=(root / "source.log").open("w"),
                    stderr=subprocess.STDOUT,
                )
                for _ in range(1200):
                    if (root / "source.json").exists():
                        break
                    assert server.poll() is None, "Source service failed"
                    time.sleep(0.1)
                source = json.loads((root / "source.json").read_text())
                assert source["status"] == "ready" and source["manifest_sha256"] == pin
                result["source_pid"] = server.pid
            phase_start = time.monotonic()
            with (root / "gpu.log").open("w") as log:
                child_env = dict(os.environ)
                child_env["BC250_D42_LABEL"] = label
                child_env["BC250_D05_CANDIDATE_TOOL"] = str(tools)
                child = subprocess.Popen(
                    command,
                    cwd=Path("/home/nik/bc250work/tools/distributed-d42"),
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
                            raise TimeoutError("D42 audit deadline")
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
                            raise TimeoutError("D42 audit deadline")
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
            if server is not None:
                server.terminate()
                server.wait(timeout=10)
                assert server.returncode == 0, "Source service failed"
                source = json.loads((root / "source.json").read_text())
                assert source["status"] == "stopped"
                result["reference_source"] = source
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
