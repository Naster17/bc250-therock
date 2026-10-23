#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Acknowledge .105's promoted r3 boot only after cooling, SSH and GPU references."""

import json
import os
import struct
import subprocess
import time
from pathlib import Path

from bc250_torch import runtime_lock
from migrate_reference_host import digest

WS = Path("/var/lib/bc250-stack/host105")
TOOLS = Path(__file__).resolve().parent
KREL = "7.2.10-bc250rocm-r3"
RELEASE = Path("/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105")


def read(path: Path) -> str:
    return path.read_text().strip()


def note_id(data: bytes) -> str | None:
    offset = 0
    while offset + 12 <= len(data):
        namesize, size, kind = struct.unpack_from("<III", data, offset)
        offset += 12
        name = data[offset : offset + namesize].rstrip(b"\0")
        offset += (namesize + 3) & ~3
        value = data[offset : offset + size]
        offset += (size + 3) & ~3
        if name == b"GNU" and kind == 3:
            return value.hex()
    return None


def main() -> None:
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    if (
        os.uname().release != KREL
        or "bc250.host105=promoted" not in read(Path("/proc/cmdline")).split()
    ):
        print("PROMOTED_KERNEL_READINESS_SKIPPED", flush=True)
        return
    boot = read(Path("/proc/sys/kernel/random/boot_id"))
    directory = WS / ("promoted-boot-" + boot)
    directory.mkdir()
    guard = Path("/run/bc250-r3-guard")
    state = json.loads((guard / "state.json").read_text())
    assert (
        state["status"] == "armed" and state["boot_id"] == boot and not state["dry_run"]
    )
    assert (
        state["phase"] == "boot"
        and state["deadline_monotonic"] - time.monotonic() > 150
    )
    variables = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            ["grub-editenv", "/boot/grub/grubenv", "list"], text=True
        ).splitlines()
        if "=" in line
    )
    assert variables["next_entry"] == "bc250-rocm-r2-kernel"
    for service in ("sshd", "watchdog", "bc250-fan-curve"):
        subprocess.run(["rc-service", service, "status"], check=True)
    watchdog = Path("/sys/class/watchdog/watchdog0")
    assert read(watchdog / "state") == "active" and read(watchdog / "nowayout") == "1"
    assert read(watchdog / "timeout") == "120"
    assert read(watchdog / "identity") == "SP5100 TCO timer"
    feeder = int(read(Path("/run/watchdog.pid")))
    device = Path("/dev/watchdog0").stat().st_rdev
    assert any(
        fd.is_char_device() and fd.stat().st_rdev == device
        for fd in Path(f"/proc/{feeder}/fd").iterdir()
    )
    hwmons = list(Path("/sys/class/hwmon").glob("hwmon*"))
    fan = next(p for p in hwmons if read(p / "name") == "nct6686")
    cpu = next(p for p in hwmons if read(p / "name") == "k10temp")
    assert int(read(fan / "fan2_input")) > 500 and int(read(fan / "pwm2")) > 0
    assert (
        read(fan / "pwm2_enable") == "1" and 0 < int(read(cpu / "temp1_input")) < 80000
    )
    assert all(Path("/sys/module", name).exists() for name in ("r8169", "nct6687"))
    with runtime_lock(
        Path("/opt/bc250-stack/migration-control/.runtime.lock"), exclusive=True
    ):
        module = Path("/lib/modules") / KREL / "updates/amdgpu.ko.gz"
        assert (
            digest(module)
            == "6ebd8393a9c7d25cd810fd89d4ef5defa187980cae64da0d37ee0d975715e8d1"
        )
        assert not Path("/sys/module/amdgpu").exists()
        before = subprocess.check_output(["dmesg"])
        (directory / "kernel-before-load.log").write_bytes(before)
        subprocess.run(["modprobe", "amdgpu"], check=True, timeout=60)
        assert (
            note_id(Path("/sys/module/amdgpu/notes/.note.gnu.build-id").read_bytes())
            == "b97bca56793aecea35967385d0c68d4186ecd121"
        )
        log = subprocess.check_output(["dmesg"])
        (directory / "kernel-after-load.log").write_bytes(log)
        assert not int(read(Path("/proc/sys/kernel/tainted"))) & (1 << 9)
        delta = log[len(before) :].decode(errors="replace")
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
        # A bounded subprocess lets the existing 240s boot deadline retain control.
        subprocess.run(
            [
                "python3",
                str(TOOLS / "qualify_migrated_host.py"),
                "--release",
                str(RELEASE),
                "--core-only",
                "--output",
                str(directory / "core"),
            ],
            check=True,
            timeout=90,
        )
    assert time.monotonic() < state["deadline_monotonic"] - 10
    # Keep the fallback armed until both the boot guardian and target references pass.
    (guard / "ack.json").write_text(
        json.dumps({key: state[key] for key in ("boot_id", "nonce", "phase")})
    )
    for _ in range(25):
        if json.loads((guard / "state.json").read_text())["status"] == "acknowledged":
            break
        time.sleep(0.2)
    assert json.loads((guard / "state.json").read_text())["status"] == "acknowledged"
    subprocess.run(
        ["grub-editenv", "/boot/grub/grubenv", "unset", "next_entry"], check=True
    )
    result = {
        "status": "passed",
        "kernel": KREL,
        "boot_id": boot,
        "core_checks": 7,
        "saved_fallback_before_readiness": variables["next_entry"],
        "watchdog_identity": read(watchdog / "identity"),
        "fan_rpm": int(read(fan / "fan2_input")),
        "source_sha256": digest(Path(__file__)),
        "module_build_id": "b97bca56793aecea35967385d0c68d4186ecd121",
        "fallback": "r2 selected before every r3 entry; cleared only after successful readiness",
        "limits": "Bounded boot readiness; arbitrary early hardware hangs remain outside proof",
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    os.sync()
    print("PROMOTED_KERNEL_READY", boot, flush=True)


if __name__ == "__main__":
    main()
