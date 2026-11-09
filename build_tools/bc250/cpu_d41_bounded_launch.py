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
import bc250_torch as launcher

run = launcher.run_process


def bounded(command):
    assert command[:2] == ["podman", "run"]
    # This is one CPU process, with neither rendezvous nor an elastic agent.
    # Remove both native device mappings before starting the container.
    removed = []
    while "--device" in command:
        index = command.index("--device")
        removed.append(command[index + 1])
        del command[index : index + 2]
    assert removed == ["/dev/kfd", "/dev/dri/renderD128"]
    command[2:2] = [
        "--label=bc250.d41=" + os.environ["BC250_D41_LABEL"],
        "--cpus=4",
        "--cpuset-cpus=0-3",
        "--memory=11g",
        "--memory-swap=12g",
        "--pids-limit=512",
    ]
    return run(command)


launcher.run_process = bounded
launcher.main()
