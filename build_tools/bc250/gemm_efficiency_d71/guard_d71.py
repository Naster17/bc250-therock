#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D71 guardian: one owned GEMM-efficiency phase under runtime lock + recovery
guard. Byte-identical logic to the qualified D69/D70 guardians (D67/L01 pattern
with the surviving distributed-d09 preflight): SH runtime lock, 600 s
boot-guard with nonce-matched ack, 460 s bounded child, kernel-log/registry
identity, foreign-container refusal, optional worker audit. Root only; the
ordinary launcher still runs the container as the recorded user.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, "/home/nik/bc250work/tools/distributed-d09")
from guard_distributed_d09_on_host import preflight  # noqa: E402

REQUIRED_KEYS = {"mode", "label", "argv", "rank", "evidence", "cwd"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    assert os.geteuid() == 0
    config = json.loads(args.config.read_text())
    assert REQUIRED_KEYS <= set(config), REQUIRED_KEYS - set(config)
    preflight()
    rank = 0 if os.uname().nodename == "bc250" else 1
    assert config["rank"] == rank
    root = Path(config["evidence"])
    root.mkdir(parents=True, exist_ok=False)
    os.chown(root, 1000, 1000)
    before = subprocess.check_output(["dmesg"])
    (root / "kernel-before.log").write_bytes(before)
    registry = Path("/opt/bc250-stack/distributed/d01.json").read_bytes()
    result = {
        "status": "running",
        "guardian_pid": os.getpid(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "kernel": os.uname().release,
        "config": config,
    }
    lock = Path(
        "/opt/bc250-stack/current/.runtime.lock"
        if rank == 0
        else "/opt/bc250-stack/migration-control/.runtime.lock"
    )
    guard = None
    state = None
    child = None

    def containers():
        return subprocess.check_output(
            ["podman", "ps", "-q", "--filter", "label=bc250.lora=" + config["label"]],
            text=True,
        ).split()

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
                    config["label"],
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
                child = subprocess.Popen(
                    [
                        "python3",
                        str(Path(__file__).with_name("bounded_launch.py")),
                        str(args.config),
                    ],
                    cwd=config["cwd"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                result["controller_pid"] = child.pid
                (root / "running.json").write_text(json.dumps(result, indent=2))
                try:
                    code = child.wait(timeout=460)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=15)
                    raise
            result["exit_code"] = code
            assert code == 0, code
            if config.get("audit"):
                report = Path(config["audit"])
                result["worker"] = json.loads(report.read_text())
                assert result["worker"]["rows"]
            text = (root / "gpu.log").read_text()
            assert not any(
                x in text for x in ["DistNetworkError", "Error waiting on exit barrier"]
            )
            result["status"] = "passed"
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            for container in containers():
                subprocess.run(["podman", "stop", "--time", 10, container], check=False)
            assert not containers(), "Owned container remains; recovery stays armed"
            after = subprocess.check_output(["dmesg"])
            (root / "kernel-after.log").write_bytes(after)
            result["kernel_log_unchanged"] = before == after
            result["registry_unchanged"] = (
                registry == Path("/opt/bc250-stack/distributed/d01.json").read_bytes()
            )
            result["foreign_containers"] = subprocess.check_output(
                ["podman", "ps", "-q"], text=True
            ).split()
            if (
                not result["kernel_log_unchanged"]
                or not result["registry_unchanged"]
                or result["foreign_containers"]
            ):
                result["status"] = "failed"
            if state:
                # The guard only accepts its exact nonce/boot_id/phase; a plain
                # marker is NOT an ack and an unacked deadline reboots the board.
                (root / "guard/ack.json").write_text(
                    json.dumps({k: state[k] for k in ["nonce", "boot_id", "phase"]})
                )
                guard.wait(timeout=10)
            (root / "final.json").write_text(json.dumps(result, indent=2))
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
