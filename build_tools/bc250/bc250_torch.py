#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run Python or an interactive shell in the active BC-250 Torch release."""

import json
import os
from pathlib import Path
import sys


def prepare_python_arguments(
    arguments: list[str], workspace: Path
) -> tuple[list[str], list[tuple[Path, str]]]:
    """Map a host script into the container while retaining the caller's cwd."""
    arguments = arguments.copy()
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("-c", "-m", "-", "-h", "--help", "-V", "--version"):
            return arguments, []
        if argument == "--":
            index += 1
            break
        if argument in ("-W", "-X", "--check-hash-based-pycs"):
            index += 2
        elif argument.startswith("-"):
            index += 1
        else:
            break
    if index >= len(arguments):
        return arguments, []
    script = Path(arguments[index]).expanduser()
    if not script.is_absolute():
        script = workspace / script
    script = script.resolve(strict=True)
    if not script.is_file():
        raise ValueError(f"Python script is not a file: {script}")
    if script.is_relative_to(workspace):
        arguments[index] = str(
            Path("/workspace/project") / script.relative_to(workspace)
        )
        return arguments, []
    arguments[index] = str(Path("/workspace/script") / script.name)
    return arguments, [(script.parent, "/workspace/script")]


def main() -> None:
    workspace = Path.cwd().resolve()
    arguments = sys.argv[1:]
    shell = arguments == ["--shell"]
    python_arguments, script_mounts = (
        (arguments, []) if shell else prepare_python_arguments(arguments, workspace)
    )
    if os.geteuid() != 0:
        os.execvp("doas", ["doas", str(Path(__file__).resolve()), *sys.argv[1:]])
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    if not manifest["validated"]:
        raise RuntimeError("The selected BC-250 release has not passed validation")
    command = ["bash"] if shell else ["python", *python_arguments]
    podman = ["podman", "run", "--rm", "--pull=never", "--network", "host", "-i"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        podman.append("-t")
    for directory, destination in script_mounts:
        podman += ["-v", f"{directory}:{destination}:rw"]
    podman += [
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri/renderD128",
        "--user",
        f"{manifest['uid']}:{manifest['gid']}",
        "--group-add",
        str(Path("/dev/kfd").stat().st_gid),
        "--group-add",
        str(Path("/dev/dri/renderD128").stat().st_gid),
        "-v",
        f"{release / 'rocm'}:/workspace/build/dist/rocm:ro",
        "-v",
        f"{release / 'torch'}:/workspace/pytorch-work:ro",
        "-v",
        f"{release / 'cache'}:/workspace/user-cache:rw",
        "-v",
        f"{workspace}:/workspace/project:rw",
        "-w",
        "/workspace/project",
        "-e",
        "ROCM_PATH=/workspace/build/dist/rocm",
        "-e",
        "HIP_PATH=/workspace/build/dist/rocm",
        "-e",
        "LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib",
        "-e",
        "PATH=/workspace/build/dist/rocm/bin:/workspace/build/dist/rocm/lib/llvm/bin:/opt/torch-venv/bin:/usr/local/therock-tools/bin:/usr/bin:/bin",
        "-e",
        "HOME=/workspace/user-cache/home",
        "-e",
        "HSA_ENABLE_SDMA=0",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "TRITON_CACHE_DIR=/workspace/user-cache/triton",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/workspace/user-cache/inductor",
        "-e",
        "TORCHINDUCTOR_COMPILE_THREADS=4",
        manifest["image"],
        *command,
    ]
    os.execvp(podman[0], podman)


if __name__ == "__main__":
    main()
