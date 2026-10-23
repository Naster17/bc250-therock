#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run an explicit prefix-update command while excluding bc250-torch workloads."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from bc250_torch import runtime_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release", type=Path, default=Path("/opt/bc250-stack/current")
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Provide a command after --")
    if os.geteuid() != 0:
        raise RuntimeError("Run through doas on the host")
    release = args.release.resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    if manifest.get("rccl_single_profile"):
        raise RuntimeError(
            "Update the dependent RCCL profile through its daily prefix lock"
        )
    if not manifest.get("mutable", False):
        raise RuntimeError("Only mutable developer prefixes can be updated")
    with runtime_lock(release / ".runtime.lock", exclusive=True):
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
