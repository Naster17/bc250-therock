#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Test-only resource adapter around the installed pinned distributed launcher."""
import os
from pathlib import Path
import sys

assert os.geteuid() == 0
sys.path.insert(0, "/opt/bc250-stack/tools/distributed-d01")
import bc250_distributed as launcher

run = launcher.run_process


def bounded(command):
    assert command[:2] == ["podman", "run"]
    command[2:2] = [
        "--cpus=4",
        "--cpuset-cpus=0-3",
        "--memory=10g",
        "--memory-swap=12g",
        "--pids-limit=512",
    ]
    return run(command)


launcher.run_process = bounded
launcher.main()
