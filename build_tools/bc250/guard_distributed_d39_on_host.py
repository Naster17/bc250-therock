#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded D39 phase guardian; preserves user processes and installed profiles."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from docker_preflight import docker_containers
from board_preflight import reservation
from artifact_d35 import load_manifest, admit_storage
from capacity_plan_d35 import CONFIG, PARAMETERS
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
    p.add_argument("--audit", choices=["legacy", "reference"], required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    a.mode = "fsdp"
    a.phase = "qualify"
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    assert a.label.replace("-", "").isalnum()
    preflight()
    rank = 0 if os.uname().nodename == "bc250" else 1
    series = Path("/home/nik/bc250work/evidence") / "distributed-d39-r1"
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
        audit=a.audit,
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
    label = "d39-r1-" + a.label

    def owned_containers():
        return subprocess.check_output(
            ["podman", "ps", "-q", "--filter", "label=bc250.d39=" + label], text=True
        ).split()

    command = [
        "python3",
        "/home/nik/bc250work/tools/distributed-d39/distributed_d39_bounded_launch.py",
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
        "/home/nik/bc250work/tools/distributed-d39/distributed_d39_units.py",
        "--audit",
        a.audit,
        "--mode",
        a.mode,
        "--phase",
        a.phase,
        "--output",
        str(root / "rank.json"),
    ]
    result["command"] = command
    result["registry_sha256"] = hashlib.sha256(registry_before).hexdigest()
    with reservation("d39-" + a.label), lock.open("r") as fd:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert registry.read_bytes() == registry_before
        reference = Path(
            "/home/nik/bc250work/evidence/distributed-d35-r1/cpu-reference"
        )
        prior = json.loads((reference / "final.json").read_text())
        assert (
            prior["status"] == "passed"
            and prior["kernel_log_unchanged"]
            and prior["registry_unchanged"]
        )
        assert (
            prior["registry_sha256"] == result["registry_sha256"]
            and prior["agent_shutdown_clean"]
        )
        assert (
            json.loads((reference / "guard/state.json").read_text())["status"]
            == "acknowledged"
        )
        reference_path = reference / "reference/manifest.json"
        pin = prior["worker"]["manifest_sha256"]
        manifest = load_manifest(reference_path, pin, CONFIG, PARAMETERS)
        assert manifest["sources"] == prior["worker"]["sources"]
        result["CPU_artifact_manifest_sha256"] = pin
        command += ["--reference", str(reference_path), "--reference-sha256", pin]
        free = shutil.disk_usage(root).free
        if a.audit == "legacy":
            result["storage"] = admit_storage(free, PARAMETERS, producing=False)
        else:
            result["storage"] = admit_storage(free, 0, producing=False)
        try:
            guarddir = root / "guard"
            guard = subprocess.Popen(
                [
                    "python3",
                    "/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py",
                    "--release",
                    os.uname().release,
                    "--phase",
                    "distributed-d39-" + a.label,
                    "--timeout",
                    "600",
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
                child_env["BC250_D39_LABEL"] = label
                child_env["BC250_D05_CANDIDATE_TOOL"] = str(tools)
                child = subprocess.Popen(
                    command,
                    cwd=Path("/home/nik/bc250work/tools/distributed-d39"),
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
                try:
                    code = child.wait(timeout=460)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=15)
                    raise
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
