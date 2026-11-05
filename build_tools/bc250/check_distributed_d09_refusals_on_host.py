#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual CLI refusals before container/device execution, preserving all runtime data."""
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    assert os.geteuid() == 0
    root = Path("/home/nik/bc250work/evidence/distributed-d09-r1")
    tool = Path("/opt/bc250-stack/tools/distributed-d09/bc250_distributed.py")
    script = Path(
        "/home/nik/bc250work/tools/distributed-d09/distributed_d09_consumer.py"
    )
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    invalid = [
        ["--env", "LD_LIBRARY_PATH=/tmp"],
        ["--env", "NCCL_PROTO=LL"],
        ["--env", "NCCL_SOCKET_IFNAME=lo"],
        ["--env", "WORLD_SIZE=1"],
        ["--env", "HSA_OVERRIDE_GFX_VERSION=10.3.0"],
        ["--env", "LD_PRELOAD=/tmp/none"],
        ["--mount", "/usr"],
        ["--mount", str(tool.parent)],
        ["--shm-size", "0"],
        ["--runtime-profile", "daily"],
        ["--shell"],
        ["--master-port", "1"],
    ]
    records = []
    for args in invalid:
        command = ["python3", str(tool), "--node-rank", "0", *args, str(script)]
        run = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert run.returncode != 0, command
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        records.append(dict(arguments=args, exit=run.returncode, stderr=run.stderr))
    (root / "CLI-refusals.json").write_text(
        json.dumps(dict(status="passed", cases=records, containers_started=0), indent=2)
        + "\n"
    )
    print("12 actual CLI refusals; no containers started")


if __name__ == "__main__":
    main()
