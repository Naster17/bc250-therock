#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy, validate, and activate a versioned BC-250 ROCm/Torch runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def run(command: list[str], log: Path | None = None) -> str:
    if log is not None:
        with log.open("w") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        return ""
    return subprocess.check_output(command, text=True).strip()


def switch_link(link: Path, target: Path) -> None:
    temporary = link.with_name(f".{link.name}-{os.getpid()}")
    temporary.symlink_to(target)
    os.replace(temporary, link)


def deploy(workspace: Path, release_name: str) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("Run deployment as root")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_name):
        raise ValueError("Invalid release name")
    if Path("/sys/class/watchdog/watchdog0/state").read_text().strip() != "active":
        raise RuntimeError("Hardware watchdog is not active")
    for name in (
        "torch-validation",
        "torch-compile-validation",
        "ml-validation",
        "primitives-validation",
        "triton-build",
    ):
        if (workspace / "logs" / f"{name}.exit").read_text().strip() != "0":
            raise RuntimeError(f"Required gate has not passed: {name}")
    wheels = list((workspace / "wheels").glob("triton-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one Triton wheel, found {wheels}")
    wheel = wheels[0]
    with wheel.open("rb") as stream:
        wheel_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    base = Path("/opt/bc250-stack")
    release = base / "releases" / release_name
    release.mkdir(parents=True, exist_ok=False)
    validation = release / "validation"
    validation.mkdir()
    torch_image = (workspace / "logs/torch-runtime-image-id").read_text().strip()
    container = "bc250-runtime-integration"
    exists = (
        subprocess.run(["podman", "container", "exists", container]).returncode == 0
    )
    if exists:
        if (
            run(["podman", "inspect", "--format", "{{.State.Running}}", container])
            != "false"
        ):
            raise RuntimeError("Runtime integration container is still running")
        run(["podman", "rm", container])
    run(
        [
            "podman",
            "run",
            "--name",
            container,
            "--network",
            "none",
            "-v",
            f"{workspace / 'wheels'}:/workspace/wheels:ro",
            torch_image,
            "python",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            f"/workspace/wheels/{wheel.name}",
        ],
        validation / "triton-install.log",
    )
    run(
        ["podman", "commit", container, f"localhost/bc250-stack:{release_name}"],
        validation / "image-commit.log",
    )
    image = run(
        [
            "podman",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            f"localhost/bc250-stack:{release_name}",
        ]
    )
    run(["rsync", "-a", f"{workspace / 'build/dist/rocm'}/", f"{release / 'rocm'}/"])
    run(
        [
            "rsync",
            "-a",
            "--exclude=/build/",
            "--exclude=/dist/",
            "--exclude=.git",
            f"{workspace / 'pytorch-work'}/",
            f"{release / 'torch'}/",
        ]
    )
    ownership = workspace.stat()
    for directory in ("home", "triton", "inductor", "hypothesis"):
        path = release / "cache" / directory
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, ownership.st_uid, ownership.st_gid)
    os.chown(release / "cache", ownership.st_uid, ownership.st_gid)
    manifest = {
        "release": release_name,
        "image": image,
        "uid": ownership.st_uid,
        "gid": ownership.st_gid,
        "therock_revision": (workspace / "source/GIT_REVISION").read_text().strip(),
        "torch_revision": (workspace / "pytorch-work/GIT_REVISION").read_text().strip(),
        "triton_revision": (workspace / "triton-work/GIT_REVISION").read_text().strip(),
        "triton_wheel_sha256": wheel_hash,
        "sdma": False,
        "validated": False,
    }
    (release / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    common = [
        "podman",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri/renderD128",
        "--user",
        f"{ownership.st_uid}:{ownership.st_gid}",
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
        f"{workspace / 'helpers'}:/workspace/helpers:ro",
        "-v",
        f"{workspace / 'triton-work'}:/workspace/triton-work:ro",
        "-v",
        f"{validation}:/workspace/logs:rw",
        "-e",
        "ROCM_PATH=/workspace/build/dist/rocm",
        "-e",
        "HIP_PATH=/workspace/build/dist/rocm",
        "-e",
        "LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib",
        "-e",
        "HOME=/workspace/user-cache/home",
        "-e",
        "HSA_ENABLE_SDMA=0",
        "-e",
        "OMP_NUM_THREADS=4",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "HYPOTHESIS_STORAGE_DIRECTORY=/workspace/user-cache/hypothesis",
        "-e",
        "TRITON_CACHE_DIR=/workspace/user-cache/triton",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/workspace/user-cache/inductor",
        "-e",
        "TORCHINDUCTOR_COMPILE_THREADS=4",
    ]
    os.chown(validation, ownership.st_uid, ownership.st_gid)
    run(["dmesg"], validation / "kernel-before.log")
    for name, script in (
        ("eager", "validate_torch.py"),
        ("compiler", "validate_torch_compile.py"),
    ):
        arguments = ["--output", f"/workspace/logs/{name}.json"]
        if name == "compiler":
            arguments += ["--triton-source", "/workspace/triton-work"]
        run(
            common
            + [
                image,
                "timeout",
                "--kill-after=10s",
                "900s",
                "python",
                f"/workspace/helpers/{script}",
                *arguments,
            ],
            validation / f"{name}.log",
        )
        (validation / f"{name}.exit").write_text("0\n")
    run(["dmesg"], validation / "kernel-after.log")
    manifest["validated"] = True
    (release / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    launcher = Path(__file__).with_name("bc250_torch.py")
    installed_launcher = Path("/usr/local/bin/bc250-torch")
    installed_launcher.parent.mkdir(parents=True, exist_ok=True)
    temporary_launcher = installed_launcher.with_name(f".bc250-torch-{os.getpid()}")
    shutil.copy2(launcher, temporary_launcher)
    temporary_launcher.chmod(0o755)
    os.replace(temporary_launcher, installed_launcher)
    current = base / "current"
    if current.is_symlink():
        switch_link(base / "previous", current.resolve(strict=True))
    elif current.exists():
        raise RuntimeError("Current release path is not a symlink")
    switch_link(current, release)
    print(f"BC250_RELEASE_ACTIVATED {release}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    deploy(args.workspace.resolve(), args.release)


if __name__ == "__main__":
    main()
