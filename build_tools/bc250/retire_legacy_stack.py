#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retire only inspected .105 legacy stack paths after target/recovery qualification."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from migrate_reference_host import digest

WS = Path("/home/nik/bc250work/migration-105-20261023")
BASE = Path("/home/nik/bc250work")
SOURCE_NAMES = (
    "rocm-libraries-gfx1013",
    "alpine-rocm-7.1",
    "alpine-rocm-aports",
    "rocm-cmake-7.2.4",
    "pytorch-build-dist",
    "pytorch-build-gfx1013",
    "bc250-rocm-profile",
    "bc250-rocm",
    "bc250-gfx1013-fix",
    "bc250-vulkan-v33",
    "research-handoff",
    "third-party",
    "apkbuild",
)
GENERATED_NAMES = (
    "distro",
    "pytorch-venv",
    "python-dev",
    "rocm-rt-libs",
    "rocm-smi-prefix",
    "openmp-prefix",
    "hipdnn-deps-prefix",
    "syslibs-fallback",
    "rocm-headers-gfx1013",
    "hipblas-build-gfx1013",
    "hipblaslt-build-gfx1013",
    "hipdnn-build-gfx1013",
    "hipfft-build-gfx1013",
    "miopen-build-gfx1013",
    "rocalution-build-gfx1013",
    "rocblas-build-gfx1013",
    "rocfft-build-gfx1013",
    "rocrand-build-gfx1013",
    "rocsolver-build-gfx1013",
    "rocsparse-build-gfx1013",
    "roctracer-build-gfx1013",
    "rpp-build-gfx1013",
)
CONTAINERS = (
    "bc250-shell-probe",
    "bc250-rocm-stable-r1",
    "bc250-dist-fix",
    "bc250-dist-keep2",
    "bc250-dist-repack",
)
IMAGES = (
    "6a30463535e5",
    "bae305af904c",
    "2ff377d288ab",
    "cd9f0f9bd955",
    "ee46e00a52a6",
    "750be6bbacd6",
    "3168a3a7a790",
    "494e7120a3d8",
)


def main() -> None:
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    assert os.uname().release == "7.2.10-bc250rocm-r3"
    assert (
        json.loads((WS / "qualification-r3/final.json").read_text())["status"]
        == "passed"
    )
    assert (
        json.loads((WS / "legacy-archive-receipt.json").read_text())["status"]
        == "passed"
    )
    assert (
        json.loads((WS / "recovery-validation.json").read_text())["status"] == "passed"
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    boot_report = (
        Path("/var/lib/bc250-stack/host105") / ("promoted-boot-" + boot) / "result.json"
    )
    if not boot_report.exists():
        boot_report = WS / ("promoted-boot-" + boot) / "result.json"
    assert json.loads(boot_report.read_text())["status"] == "passed"
    assert (
        json.loads((WS / "legacy-source-compare.json").read_text())["status"]
        == "passed"
    )
    fallback = json.loads((WS / "before.json").read_text())["fallback_hashes"]
    assert all(digest(Path(name)) == sha for name, sha in fallback.items())
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    assert release == Path("/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105")
    paths = [BASE / name for name in SOURCE_NAMES + GENERATED_NAMES] + [
        Path("/home/nik/bc250-stack")
    ]
    mounts = [
        Path(line.split()[4])
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
    ]
    for path in paths:
        assert not path.is_symlink() and not any(m.is_relative_to(path) for m in mounts)
        assert not release.is_relative_to(path)
    # Refuse retirement if a live executable is still supplied by one of these trees.
    for process in Path("/proc").glob("[0-9]*"):
        try:
            executable = (process / "exe").resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        assert not any(executable.is_relative_to(path) for path in paths), str(
            executable
        )
    report = {
        "status": "running",
        "deleted_paths": [],
        "removed_containers": [],
        "removed_images": [],
        "disk_free_before": shutil.disk_usage("/").free,
        "fallback_hashes": fallback,
        "retained": [
            "r2 and distro boot fallbacks/modules",
            "native cooling/governor/core-unlock/SMI utilities",
            "Docker storage and unrelated base images",
            "user applications/models and home caches",
        ],
    }

    def save() -> None:
        (WS / "retirement.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for name in CONTAINERS:
        if (
            subprocess.run(
                ["podman", "container", "exists", name], check=False
            ).returncode
            == 0
        ):
            state = subprocess.check_output(
                ["podman", "inspect", "--format", "{{.State.Running}}", name], text=True
            ).strip()
            assert state == "false", name
            subprocess.run(["podman", "rm", name], check=True)
            report["removed_containers"].append(name)
            save()
    for image in IMAGES:
        if (
            subprocess.run(["podman", "image", "exists", image], check=False).returncode
            == 0
        ):
            subprocess.run(["podman", "rmi", image], check=True)
            report["removed_images"].append(image)
            save()
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
            report["deleted_paths"].append(str(path))
            save()
    assert all(digest(Path(name)) == sha for name, sha in fallback.items())
    report.update(status="passed", disk_free_after=shutil.disk_usage("/").free)
    save()
    os.sync()
    print(
        "LEGACY_STACK_RETIRED",
        report["disk_free_after"] - report["disk_free_before"],
        flush=True,
    )


if __name__ == "__main__":
    main()
