#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Test-only resource adapter around the installed pinned distributed launcher."""
import os
from pathlib import Path
import sys

assert os.geteuid() == 0
sys.path.insert(
    0,
    os.environ.get(
        "BC250_D05_CANDIDATE_TOOL",
        str(Path("/usr/local/bin/bc250-distributed").resolve().parent),
    ),
)
import bc250_distributed as launcher

run = launcher.run_process


def bounded(command):
    assert command[:2] == ["podman", "run"]
    command[2:2] = [
        "--label=bc250.d12=" + os.environ["BC250_D12_LABEL"],
        "--cpus=4",
        "--cpuset-cpus=0-3",
        "--memory=10g",
        "--memory-swap=12g",
        "--pids-limit=512",
    ]
    return run(command)


launcher.run_process = bounded
launcher.main()
