#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Test-only resource envelope around the existing ordinary launcher."""
import os
import sys

sys.path.insert(0, "/home/nik/bc250work/rocm-10.2-gfx1013/helpers")
import bc250_torch as launcher

native = launcher.run_process


def bounded(command):
    assert command[:2] == ["podman", "run"]
    command[2:2] = [
        "--cpus=4",
        "--cpuset-cpus=0-3",
        "--memory=10g",
        "--memory-swap=12g",
        "--pids-limit=512",
        "--name=bc250-plan02-" + str(os.getpid()),
    ]
    return native(command)


launcher.run_process = bounded
launcher.main()
