#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only, allowlisted BC-250 migration/retirement inventory."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def command(*args: str) -> dict:
    result = subprocess.run(
        args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return {"exit": result.returncode, "output": result.stdout}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "kernel": os.uname().release,
        "hostname": os.uname().nodename,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cmdline": Path("/proc/cmdline").read_text().strip(),
        "disk": command("df", "-k", "/"),
        "images": command("podman", "images", "--no-trunc", "--format", "json"),
        "containers": command("podman", "ps", "-a", "--no-trunc", "--format", "json"),
        "services": command("rc-status", "-a"),
        "grubenv": command("grub-editenv", "/boot/grub/grubenv", "list"),
        "configs": {},
        "repositories": {},
        "stack_paths": {},
    }
    for name in (
        "/etc/default/grub",
        "/etc/grub.d/40_custom",
        "/etc/conf.d/watchdog",
        "/etc/init.d/bc250-r3-bootguard",
        "/etc/modules-load.d/nct6687.conf",
    ):
        path = Path(name)
        if path.exists():
            result["configs"][name] = path.read_text()
    root = Path("/home/nik/bc250work")
    # Only stack directories: user applications and arbitrary home trees excluded.
    names = (
        "alpine-rocm",
        "alpine-rocm-7.1",
        "alpine-rocm-aports",
        "apkbuild",
        "distro",
        "pytorch-build-dist",
        "pytorch-build-gfx1013",
        "pytorch-venv",
        "rocm-libraries-gfx1013",
        "bc250-rocm-lab",
        "bc250-rocm-profile",
        "rocm-rt-libs",
        "rocm-smi-prefix",
        "rocm-cmake-7.2.4",
    )
    for name in names:
        path = root / name
        if path.exists():
            result["stack_paths"][str(path)] = command("du", "-sk", str(path))
    # Bounded depth identifies nested component repos, without reading user apps.
    for name in names:
        base = root / name
        for current, directories, files in os.walk(base):
            current = Path(current)
            if ".git" in directories or ".git" in files:
                result["repositories"][str(current)] = {
                    key: command(
                        "git",
                        "-c",
                        "safe.directory=" + str(current),
                        "-C",
                        str(current),
                        *arguments,
                    )
                    for key, arguments in {
                        "revision": ["rev-parse", "HEAD"],
                        "changes": ["status", "--short"],
                        "branches": ["branch", "-avv"],
                    }.items()
                }
            directories[:] = [
                d for d in directories if d not in (".git", "build", ".cache")
            ]
            if len(current.relative_to(base).parts) >= 4:
                directories[:] = []
    result["fallback_hashes"] = {}
    for name in ("vmlinuz-7.1.5-bc250rocm-r2", "initramfs-7.1.5-bc250rocm-r2"):
        path = root / "bc250-rocm-lab" / name
        with path.open("rb") as stream:
            result["fallback_hashes"][str(path)] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("MIGRATION_AUDIT_SAVED", args.output)


if __name__ == "__main__":
    main()
