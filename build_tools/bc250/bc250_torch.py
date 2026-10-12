#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run Python or an interactive shell in the active BC-250 Torch release."""

import json
import os
from pathlib import Path
import sys


def main() -> None:
    if os.geteuid() != 0:
        os.execvp("doas", ["doas", str(Path(__file__).resolve()), *sys.argv[1:]])
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    if not manifest["validated"]:
        raise RuntimeError("The selected BC-250 release has not passed validation")
    arguments = sys.argv[1:]
    command = ["bash"] if arguments == ["--shell"] else ["python", *arguments]
    workspace = Path.cwd().resolve()
    podman = ["podman", "run", "--rm", "--pull=never", "--network", "host", "-i"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        podman.append("-t")
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
