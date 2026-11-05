#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded D06 phase guardian; preserves user processes and installed profiles."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def preflight():
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
    p.add_argument("--mode", choices=["single", "sync", "no-sync"], required=True)
    p.add_argument("--case", choices=["small", "larger"], required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    assert a.label.replace("-", "").isalnum()
    preflight()
    rank = 0 if os.uname().nodename == "bc250" else 1
    root = Path("/home/nik/bc250work/evidence/distributed-d06-r1") / a.label
    root.mkdir(parents=True)
    os.chown(root, 1000, 1000)
    lock = Path(
        "/opt/bc250-stack/current/.runtime.lock"
        if rank == 0
        else "/opt/bc250-stack/migration-control/.runtime.lock"
    )
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    registry_before = registry.read_bytes()
    before = subprocess.check_output(["dmesg"])
    (root / "kernel-before.log").write_bytes(before)
    state = None
    guard = None
    child = None
    result = dict(status="running", rank=rank, mode=a.mode, kernel=os.uname().release)
    command = [
        "python3",
        "/home/nik/bc250work/tools/distributed-d05/distributed_d05_bounded_launch.py",
        "--node-rank",
        str(rank),
        "--master-port",
        str(a.port),
        "/home/nik/bc250work/tools/distributed-d06/distributed_d06_accumulate.py",
        "--mode",
        a.mode,
        "--case",
        a.case,
        "--output",
        str(root / "rank.json"),
    ]
    result["command"] = command
    result["registry_sha256"] = hashlib.sha256(registry_before).hexdigest()
    with lock.open("r") as fd:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            guarddir = root / "guard"
            guard = subprocess.Popen(
                [
                    "python3",
                    "/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py",
                    "--release",
                    os.uname().release,
                    "--phase",
                    "distributed-d06-" + a.label,
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
            with (root / "gpu.log").open("w") as log:
                child_env = dict(os.environ)
                child = subprocess.Popen(
                    command,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = child.wait(timeout=460)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=15)
                    raise
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
                if not subprocess.check_output(
                    ["podman", "ps", "-q"], text=True
                ).strip():
                    break
                time.sleep(0.1)
            assert not subprocess.check_output(
                ["podman", "ps", "-q"], text=True
            ).strip(), "Owned worker still running; guard deliberately remains armed"
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
