#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Copy a P2 candidate independently and test it without switching the dev launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from bc250_torch import runtime_lock


def run(command: list[str], log: Path | None = None) -> None:
    if log is None:
        subprocess.run(command, check=True)
    else:
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    assert os.geteuid() == 0
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    ws = args.workspace.resolve(strict=True)
    candidate = args.candidate.absolute()
    assert not candidate.exists(), candidate
    dev = Path("/opt/bc250-stack/current").resolve(strict=True)
    runtime = json.loads((dev / "runtime.json").read_text())
    assert runtime.get("mutable")
    assert (ws / "logs/priority2-targeted-stage.exit").read_text().strip() == "0"
    candidate.mkdir(parents=True)
    validation = candidate / "validation"
    validation.mkdir()
    required = {
        "priority2-scalar.json",
        "priority2-lt.json",
        "priority2-performance-after.json",
        "priority2-math.json",
        "priority2-primitives.json",
    }
    for name in required:
        report = json.loads((ws / "logs" / name).read_text())
        assert report["status"] == "passed", name
        if name == "priority2-math.json":
            assert len(report["checks"]) >= 149 and all(
                x["status"] == "passed" for x in report["checks"]
            )
        elif name == "priority2-primitives.json":
            assert report["cases"] >= 138
        elif name == "priority2-scalar.json":
            counts = {x["name"]: x["executed_tests"] for x in report["checks"]}
            assert (
                counts["hiptensor_scalar_smoke"] >= 166
                and counts["hipsparselt_scalar_smoke"] >= 234
            )
        elif name == "priority2-lt.json":
            counts = {x["name"]: x["executed_tests"] for x in report["checks"]}
            assert (
                counts["hipblaslt_priority1_smoke"] >= 561
                and counts["hipblaslt_scalar_smoke"] >= 403
            )
        elif name == "priority2-performance-after.json":
            assert len(report["cases"]) >= 65
        shutil.copy2(ws / "logs" / name, validation / name)
    with runtime_lock(dev / ".runtime.lock", exclusive=True):
        for source, dest, exclude in [
            (ws / "build/dist/rocm", candidate / "rocm", []),
            (
                ws / "pytorch-work",
                candidate / "torch",
                ["--exclude=/build/", "--exclude=/dist/", "--exclude=.git"],
            ),
        ]:
            run(["rsync", "-a", *exclude, str(source) + "/", str(dest) + "/"])
        checks = {}
        # Critical executable payloads must be byte-identical independent copies.
        for source_root, dest_root in [
            (ws / "build/dist/rocm", candidate / "rocm"),
            (ws / "pytorch-work", candidate / "torch"),
        ]:
            for source in source_root.rglob("*"):
                if ".so" not in source.name and source.suffix != ".kpack":
                    continue
                if source.is_symlink() or not source.is_file():
                    continue
                relative = source.relative_to(source_root)
                target = dest_root / relative
                if not target.exists():
                    continue  # excluded build intermediates
                assert not os.path.samefile(source, target), target
                with source.open("rb") as stream:
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                with target.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                assert actual == expected, target
                checks[str(target.relative_to(candidate))] = actual
        for root in (candidate / "rocm", candidate / "torch"):
            for path in root.rglob("*"):
                if path.is_symlink():
                    # Relative installation symlinks are fine; no pointer into mutable WS.
                    assert not str(path.resolve()).startswith(str(ws) + "/"), path
        manifest = {
            **runtime,
            "release": candidate.name,
            "mutable": False,
            "validated": False,
            "qualification": "independent P2 candidate; bounded GPU gate pending",
            "payload_sha256": checks,
            "created_unix": time.time(),
        }
        (candidate / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cache = candidate / "cache"
    cache.mkdir()
    for name in ("home", "inductor", "triton"):
        (cache / name).mkdir()
        os.chown(cache / name, runtime["uid"], runtime["gid"])
    os.chown(cache, runtime["uid"], runtime["gid"])
    os.chown(validation, runtime["uid"], runtime["gid"])
    report = validation / "candidate.json"
    prefix = "/workspace/build/dist/rocm"
    command = [
        "podman",
        "run",
        "--rm",
        "--network",
        "none",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri/renderD128",
        "-v",
        f'{candidate / "rocm"}:{prefix}:ro',
        "-v",
        f'{candidate / "torch"}:/workspace/pytorch-work:ro',
        "-v",
        f"{cache}:/workspace/candidate-cache:rw",
        "-v",
        f"{validation}:/workspace/validation:rw",
        "-v",
        f'{ws / "helpers"}:/workspace/helpers:ro',
        "-e",
        "HSA_ENABLE_SDMA=0",
        "-e",
        f"LD_LIBRARY_PATH={prefix}/lib:{prefix}/lib/llvm/lib",
        "-e",
        f"PATH={prefix}/bin:{prefix}/lib/llvm/bin:/opt/torch-venv/bin:/usr/local/therock-tools/bin:/usr/bin:/bin",
        "-e",
        f"ROCM_PATH={prefix}",
        "-e",
        f"HIP_PATH={prefix}",
        "-e",
        "HOME=/workspace/candidate-cache/home",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/workspace/candidate-cache/inductor",
        "-e",
        "TRITON_CACHE_DIR=/workspace/candidate-cache/triton",
        "-e",
        "TORCHINDUCTOR_COMPILE_THREADS=4",
        runtime["image"],
        "timeout",
        "--kill-after=10s",
        "240s",
        "python",
        "/workspace/helpers/validate_priority2_candidate.py",
        "--output",
        "/workspace/validation/candidate.json",
    ]
    # Exercise the independent payload with the same unprivileged UID as the launcher.
    command[2:2] = ["--user", f"{runtime['uid']}:{runtime['gid']}"]
    for group in sorted(
        {Path("/dev/kfd").stat().st_gid, Path("/dev/dri/renderD128").stat().st_gid}
    ):
        command[2:2] = ["--group-add", str(group)]
    run(command, validation / "candidate.log")
    assert json.loads(report.read_text())["status"] == "passed"
    manifest.update(
        validated=True,
        qualification="independent copies and bounded P2 GPU checks; historical crash cause unconfirmed",
    )
    (candidate / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert Path("/opt/bc250-stack/current").resolve() == dev
    print(
        "PRIORITY2_PACKAGE_PASS",
        candidate,
        len(checks),
        "independent shared-object hashes",
        flush=True,
    )


if __name__ == "__main__":
    main()
