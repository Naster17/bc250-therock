#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Arm/finalize the ordinary-user interactive authentication test without storing secrets.

The recorded ordinary-user command runs validate_ck_plugin_b0713_user.py and
writes logs/result.json (2 checks) before finish acknowledges the guard.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

LABEL = "ck-plugin-b0713-user-r1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("arm", "finish"))
    args = p.parse_args()
    logs = WS / "logs" / LABEL
    guard = Path("/run") / LABEL
    if args.action == "arm":
        preflight(LABEL)
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        logs.mkdir()
        os.chown(logs, 1000, 1000)
        (logs / "kernel-before.log").write_bytes(subprocess.check_output(["dmesg"]))
        (logs / "user.env").write_text("BC250_CK_FILE=file-value\n")
        with (logs / "guard.log").open("w") as f:
            proc = subprocess.Popen(
                [
                    "python3",
                    "/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py",
                    "--release",
                    os.uname().release,
                    "--phase",
                    LABEL,
                    "--timeout",
                    "600",
                    "--directory",
                    str(guard),
                ],
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(100):
            if (guard / "state.json").exists():
                break
            assert proc.poll() is None
            time.sleep(0.05)
        state = json.loads((guard / "state.json").read_text())
        assert state["status"] == "armed"
        atomic_json(logs / "armed.json", dict(status="armed", guard_pid=proc.pid))
    else:
        running = subprocess.check_output(["podman", "ps", "-q"], text=True).split()
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        state = json.loads((guard / "state.json").read_text())
        assert not running
        atomic_json(
            guard / "ack.json", {k: state[k] for k in ("boot_id", "nonce", "phase")}
        )
        for _ in range(100):
            state = json.loads((guard / "state.json").read_text())
            if state["status"] == "acknowledged":
                break
            time.sleep(0.05)
        result = json.loads((logs / "result.json").read_text())
        assert (
            result["status"] == "passed"
            and len(result["checks"]) == 2
            and state["status"] == "acknowledged"
        )
        assert after == (logs / "kernel-before.log").read_bytes()
        atomic_json(
            logs / "final.json",
            dict(
                status="passed",
                checks=2,
                guard=state,
                kernel_log_unchanged=True,
                running_containers=running,
            ),
        )
    print("CK_PLUGIN_B0713_USER_" + args.action.upper() + "_PASS", flush=True)


if __name__ == "__main__":
    main()
