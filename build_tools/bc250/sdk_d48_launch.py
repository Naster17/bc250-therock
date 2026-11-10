#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Private bounded SDK test adapter. Public launcher validation stays untouched."""
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path

assert os.geteuid() == 0
prefix = Path("/home/nik/bc250work/evidence/sdk-d48-build-r2/prefix")
spec = importlib.util.spec_from_loader(
    "d48_launcher", SourceFileLoader("d48_launcher", "/usr/local/bin/bc250-torch")
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
validate = launcher.validate_options
build = launcher.build_command


def candidate_validation(options):
    # Exercise this one native sentinel privately. All other shared validation
    # and protected-environment checks remain active, including public refusal.
    if options.environment.get("HIP_VISIBLE_DEVICES") != "-1":
        return validate(options)
    value = options.environment.pop("HIP_VISIBLE_DEVICES")
    try:
        validate(options)
    finally:
        options.environment["HIP_VISIBLE_DEVICES"] = value


def candidate_command(*args):
    command = build(*args)
    assert command[:2] == ["podman", "run"]
    control = os.environ.get("BC250_D48_CONTROL") == "1"
    sha = (
        "a6dc1c884c4f446e23a2afc8185b26ef9fd55b57cdd1d15324d46054b902b9c2"
        if control
        else json.loads((prefix.parent / "build.json").read_text())["library_sha256"]
    )
    command[2:2] = [
        "--label=bc250.d48=test-r3",
        "--cpus=4",
        "--cpuset-cpus=0-3",
        "--memory=4g",
        "--memory-swap=4g",
        "--pids-limit=512",
        "--ulimit=core=0:0",
        "-e",
        "BC250_D48_SDK_SHA=" + sha,
        "-e",
        "BC250_D48_CONTROL=" + str(int(control)),
    ]
    if control and os.environ.get("BC250_D48_PRELOAD_CONTROL") == "1":
        command[2:2] = [
            "-e",
            "LD_PRELOAD=/workspace/build/dist/rocm/lib/librocprofiler-sdk.so.1.4.2",
        ]
    if not control:
        command[2:2] = [
            "-v",
            f"{prefix}:/workspace/sdk-d48:ro",
        ]
        positions = [
            i for i, value in enumerate(command) if value.startswith("LD_LIBRARY_PATH=")
        ]
        assert len(positions) == 1
        i = positions[0]
        command[i] = (
            "LD_LIBRARY_PATH=/workspace/sdk-d48/lib:" + command[i].partition("=")[2]
        )
    return command


launcher.validate_options = candidate_validation
launcher.build_command = candidate_command
launcher.main()
