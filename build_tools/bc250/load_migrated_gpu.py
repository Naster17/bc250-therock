#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guard the first .105 r3 AMDGPU load and bounded packaged GPU smoke."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from migrate_reference_host import digest
from ready_migrated_kernel import note_id

WS = Path("/home/nik/bc250work/migration-105-20261023")


def main() -> None:
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    assert os.uname().release == "7.2.10-bc250rocm-r3"
    assert not Path("/sys/module/amdgpu").exists()
    directory = WS / "trial-gpu-load"
    directory.mkdir()
    guard_dir = directory / "guard"
    guard = subprocess.Popen(
        [
            sys.executable,
            "/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py",
            "--phase",
            "migration-gpu-load",
            "--timeout",
            "600",
            "--directory",
            str(guard_dir),
        ],
        stdout=(directory / "guard.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        if (guard_dir / "state.json").exists():
            break
        time.sleep(0.05)
    state = json.loads((guard_dir / "state.json").read_text())
    assert state["status"] == "armed" and not state["dry_run"]
    before = subprocess.check_output(["dmesg"])
    (directory / "kernel-before.log").write_bytes(before)
    subprocess.run(["modprobe", "amdgpu"], check=True, timeout=90)
    loaded_module = note_id(
        Path("/sys/module/amdgpu/notes/.note.gnu.build-id").read_bytes()
    )
    loaded_kernel = note_id(Path("/sys/kernel/notes").read_bytes())
    assert loaded_module == "b97bca56793aecea35967385d0c68d4186ecd121"
    assert loaded_kernel == "57f7c52d9c7ac0ac21c67d6e6efc3e08c12bbfde"
    assert (
        digest(Path("/lib/modules/7.2.10-bc250rocm-r3/updates/amdgpu.ko.gz"))
        == "6ebd8393a9c7d25cd810fd89d4ef5defa187980cae64da0d37ee0d975715e8d1"
    )
    assert not int(Path("/proc/sys/kernel/tainted").read_text()) & (1 << 9)
    after = subprocess.check_output(["dmesg"])
    (directory / "kernel-after.log").write_bytes(after)
    assert after.startswith(before)
    delta = after[len(before) :].decode(errors="replace")
    assert not any(
        word in delta
        for word in (
            "*ERROR*",
            "WARNING:",
            "BUG:",
            "GPU fault",
            "Failed to clear hpd(rx)",
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(WS / "qualify_migrated_host.py"),
            "--release",
            "/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105",
            "--core-only",
            "--output",
            str(directory / "core"),
        ],
        check=True,
        timeout=180,
    )
    (guard_dir / "ack.json").write_text(
        json.dumps({k: state[k] for k in ("boot_id", "nonce", "phase")})
    )
    assert guard.wait(timeout=5) == 0
    result = {
        "status": "passed",
        "boot_id": state["boot_id"],
        "module_build_id": loaded_module,
        "kernel_build_id": loaded_kernel,
        "core_checks": 7,
        "limits": "Driver load and seven GPU CPU-reference smoke checks; broader target suite separate",
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MIGRATED_GPU_LOAD_PASS", flush=True)


if __name__ == "__main__":
    main()
