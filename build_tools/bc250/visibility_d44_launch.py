#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Root-only D44 resource adapter; does not edit native launcher defaults."""
import importlib.util
from importlib.machinery import SourceFileLoader
import os
from pathlib import Path
import sys

assert os.geteuid() == 0
tool = Path(sys.argv.pop(1)).resolve(strict=True)
spec = importlib.util.spec_from_loader(
    "d44_launcher", SourceFileLoader("d44_launcher", str(tool))
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
original = launcher.run_process


def bounded(command):
    assert command[:2] == ["podman", "run"]
    command[2:2] = [
        "--label=bc250.d44=visibility-r1",
        "--cpus=4",
        "--cpuset-cpus=0-3",
        "--memory=4g",
        "--memory-swap=4g",
        "--pids-limit=512",
        "--ulimit=core=0:0",
    ]
    return original(command)


launcher.run_process = bounded
try:
    launcher.main()
except (ValueError, FileNotFoundError, PermissionError) as error:
    print(f"bc250-torch: {error}", file=sys.stderr)
    raise SystemExit(2)
