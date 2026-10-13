#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Seal the existing qualified release without deleting development artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


WORKSPACE_FILES = (
    "expanded-build.exit",
    "expanded-validation.exit",
    "candidate-torch-validation.exit",
    "launcher-io-validation.exit",
    "training-memory.exit",
)


def run(command: list[str], log: Path | None = None) -> str:
    if log:
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return ""
    return subprocess.check_output(command, text=True).strip()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--review-source", required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError("Finalize the local release as root")
    workspace = args.workspace.resolve()
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    runtime = json.loads((release / "runtime.json").read_text())
    if not runtime["validated"]:
        raise RuntimeError("The active runtime is not qualified")
    for name in WORKSPACE_FILES:
        if (workspace / "logs" / name).read_text().strip() != "0":
            raise RuntimeError(f"Required qualification gate did not pass: {name}")
    for name in ("eager", "compiler", "stability"):
        if (release / "validation" / f"{name}.exit").read_text().strip() != "0":
            raise RuntimeError(f"Installed release gate did not pass: {name}")
    stability = json.loads((release / "validation/stability.json").read_text())
    if stability["status"] != "passed" or stability["seconds"] < 600:
        raise RuntimeError("Installed stability evidence is incomplete")
    actual_image = run(
        ["podman", "image", "inspect", "--format", "{{.Id}}", runtime["image"]]
    )
    if actual_image != runtime["image"]:
        raise RuntimeError("Runtime image identity differs")
    launcher = Path("/usr/local/bin/bc250-torch")
    if sha256(launcher) != sha256(Path(__file__).with_name("bc250_torch.py")):
        raise RuntimeError("Installed launcher differs from the reviewed local source")
    if Path("/sys/class/watchdog/watchdog0/state").read_text().strip() != "active":
        raise RuntimeError("Watchdog keepalives are not active")
    previous = Path("/opt/bc250-stack/previous").resolve(strict=True)
    if not json.loads((previous / "runtime.json").read_text())["validated"]:
        raise RuntimeError("Fallback runtime is not validated")

    # Add independent consumer files; no ROCm/Torch files or old environments change.
    consumers = release / "python-consumers"
    if consumers.exists():
        raise RuntimeError(
            "Consumer snapshot already exists; preserve and review it before retrying"
        )
    consumers.mkdir()
    source_environment = workspace / "training-env-stable"
    source_sites = list((source_environment / "venv/lib").glob("python*/site-packages"))
    if len(source_sites) != 1:
        raise RuntimeError("Cannot identify the qualified training package directory")
    run(
        [
            "rsync",
            "-a",
            "--chown=0:0",
            f"{source_sites[0]}/",
            f"{consumers / 'site-packages'}/",
        ]
    )
    run(["chmod", "-R", "go-w", str(consumers)])
    shutil.copy2(source_environment / "packages.json", consumers / "packages.json")
    shutil.copy2(
        source_environment / "runtime-constraints.txt",
        consumers / "runtime-constraints.txt",
    )
    (consumers / "training.env").write_text(
        "PYTHONPATH=/opt/bc250-stack/current/python-consumers/site-packages\n"
        "HF_HUB_DISABLE_XET=1\nTOKENIZERS_PARALLELISM=false\nOMP_NUM_THREADS=8\n"
    )
    review = release / "review-20261013"
    review.mkdir(exist_ok=False)
    review.chmod(0o755)
    for name in ("home", "cache", "logs"):
        path = review / name
        path.mkdir()
        os.chown(path, runtime["uid"], runtime["gid"])
    run(["dmesg"], review / "kernel-before.log")
    container = f"bc250-release-review-{int(time.time())}"
    command = [
        "podman",
        "run",
        "--name",
        container,
        "--pull=never",
        "--network",
        "none",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri/renderD128",
        "--user",
        f"{runtime['uid']}:{runtime['gid']}",
    ]
    for device in ("/dev/kfd", "/dev/dri/renderD128"):
        command += ["--group-add", str(Path(device).stat().st_gid)]
    for source, destination, mode in (
        (release / "rocm", "/workspace/build/dist/rocm", "ro"),
        (release / "torch", "/workspace/pytorch-work", "ro"),
        (consumers / "site-packages", "/workspace/release-consumers", "ro"),
        (
            Path(__file__).with_name("release_smoke.py"),
            "/workspace/release_smoke.py",
            "ro",
        ),
        (review, "/workspace/review", "rw"),
    ):
        command += ["-v", f"{source}:{destination}:{mode}"]
    for name, value in {
        "ROCM_PATH": "/workspace/build/dist/rocm",
        "HIP_PATH": "/workspace/build/dist/rocm",
        "LD_LIBRARY_PATH": "/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib",
        "PATH": "/workspace/build/dist/rocm/bin:/workspace/build/dist/rocm/lib/llvm/bin:/opt/torch-venv/bin:/usr/local/therock-tools/bin:/usr/bin:/bin",
        "HOME": "/workspace/review/home",
        "PYTHONPATH": "/workspace/release-consumers",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HSA_ENABLE_SDMA": "0",
        "OMP_NUM_THREADS": "4",
        "TRITON_CACHE_DIR": "/workspace/review/cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/workspace/review/cache/inductor",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
    }.items():
        command += ["-e", f"{name}={value}"]
    # Keep the stopped review container as evidence; do not remove/prune anything.
    run(
        command
        + [
            runtime["image"],
            "timeout",
            "-k",
            "10",
            "120",
            "python",
            "/workspace/release_smoke.py",
            "--output",
            "/workspace/review/logs/standalone.json",
        ],
        review / "standalone.log",
    )
    run(["dmesg"], review / "kernel-after.log")
    if (review / "kernel-before.log").read_bytes() != (
        review / "kernel-after.log"
    ).read_bytes():
        raise RuntimeError(
            "Kernel messages changed during review; inspect before sealing"
        )
    tools = review / "tools"
    tools.mkdir()
    for name in ("bc250_torch.py", "bc250_train.sh"):
        shutil.copy2(Path(__file__).with_name(name), tools / name)
    train_launcher = Path("/usr/local/bin/bc250-train")
    if train_launcher.exists():
        raise RuntimeError(
            "An existing training launcher must be reviewed before replacement"
        )
    temporary = train_launcher.with_name(".bc250-train-release")
    shutil.copy2(tools / "bc250_train.sh", temporary)
    temporary.chmod(0o755)
    os.replace(temporary, train_launcher)

    checksums = {}
    for tree in (
        release / "rocm",
        release / "torch/torch",
        consumers / "site-packages",
    ):
        for path in sorted(tree.rglob("*")):
            if path.is_file() and not path.is_symlink():
                checksums[str(path.relative_to(release))] = sha256(path)
    (review / "runtime-sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")
    manifest = {
        "label": "BC-250 ROCm 10.2 daily-use snapshot 2026-10-13",
        "status": "reviewed-local-preview",
        "release": str(release),
        "runtime": runtime,
        "review_source_revision": args.review_source,
        "launcher_sha256": sha256(launcher),
        "training_launcher_sha256": sha256(train_launcher),
        "consumer_packages": json.loads((consumers / "packages.json").read_text()),
        "fallback_release": str(previous),
        "review_container": container,
        "standalone_smoke": json.loads((review / "logs/standalone.json").read_text()),
        "kernel_log_unchanged": True,
        "checksummed_files": len(checksums),
        "preservation": "All development sources, builds, caches, images, logs, checkpoints and prior releases retained",
        "development_resume_target": "2026-10-18; manual continuation only",
    }
    (review / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"DAILY_USE_RELEASE_REVIEW_PASS {review / 'release.json'}", flush=True)


if __name__ == "__main__":
    main()
