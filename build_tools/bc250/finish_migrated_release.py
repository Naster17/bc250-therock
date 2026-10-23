#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the independent .105 payload and remove only this migration's transfers."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from migrate_reference_host import digest, inventory
from ready_migrated_kernel import note_id

WS = Path("/home/nik/bc250work/migration-105-20261023")
RELEASE = Path("/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105")


def main() -> None:
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    assert json.loads((WS / "retirement.json").read_text())["status"] == "passed"
    assert Path("/opt/bc250-stack/current").resolve() == RELEASE
    assert {"cpu", "memory", "pids"}.issubset(
        Path("/sys/fs/cgroup/cgroup.controllers").read_text().split()
    )
    expected = json.loads((WS / "source-payload-files-locked.json").read_text())
    actual = {name: inventory(RELEASE / name) for name in ("rocm", "torch")}
    assert actual == expected, "Native/source payload changed after retirement"
    manifest = json.loads((RELEASE / "runtime.json").read_text())
    assert (
        subprocess.check_output(
            ["podman", "image", "inspect", "--format", "{{.Id}}", manifest["image"]],
            text=True,
        ).strip()
        == manifest["image"]
    )
    for name, sha in manifest["operational_boot_tools"]["sha256"].items():
        assert digest(RELEASE / "healthchecks" / name) == sha
    free_before = shutil.disk_usage("/").free
    removed = []
    for name in (
        "runtime.tar.gz",
        "image.tar.gz",
        "image-direct.tar.gz",
        "kernel.tar.gz",
        "consumers-launchers.tar.gz",
        "initramfs-r3",
    ):
        path = WS / name
        if path.exists():
            assert path.is_file() and not path.is_symlink()
            path.unlink()
            removed.append(str(path))
    result = {
        "status": "payload_passed_final_boot_pending",
        "removed_owned_transfers": removed,
        "native_files_and_symlinks": {
            name: len(files) for name, files in actual.items()
        },
        "disk_free_before": free_before,
        "disk_free_after": shutil.disk_usage("/").free,
        "module_build_id": note_id(
            Path("/sys/module/amdgpu/notes/.note.gnu.build-id").read_bytes()
        ),
        "image": manifest["image"],
        "runtime_sha256": digest(RELEASE / "runtime.json"),
    }
    (WS / "finish.json").write_text(json.dumps(result, indent=2) + "\n")
    subprocess.run(
        [
            "python3",
            str(RELEASE / "healthchecks/qualify_migrated_host.py"),
            "--release",
            str(RELEASE),
            "--core-only",
            "--output",
            "/var/lib/bc250-stack/host105/post-retirement-core",
        ],
        check=True,
        timeout=180,
    )
    result["status"] = "passed_final_boot_pending"
    (WS / "finish.json").write_text(json.dumps(result, indent=2) + "\n")
    os.sync()
    print("MIGRATED_PAYLOAD_POST_RETIREMENT_PASS", flush=True)


if __name__ == "__main__":
    main()
