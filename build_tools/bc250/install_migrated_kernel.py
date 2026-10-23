#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install qualified r3 payload on .105, retaining r2 default; never reboot."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

WS = Path("/home/nik/bc250work/migration-105-20261023")
KWS = Path("/home/nik/bc250work/kernel-r3-7.2.10")
KREL = "7.2.10-bc250rocm-r3"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    assert not KWS.exists() and not Path("/lib/modules", KREL).exists()
    assert "GRUB_DEFAULT=bc250-rocm-r2-kernel" in Path("/etc/default/grub").read_text()
    env = run("grub-editenv", "/boot/grub/grubenv", "list")
    assert not dict(line.split("=", 1) for line in env.splitlines() if "=" in line).get(
        "next_entry"
    )
    backup = WS / "boot-backup"
    backup.mkdir()
    files = (
        "/etc/default/grub",
        "/etc/grub.d/40_custom",
        "/boot/grub/grub.cfg",
        "/boot/grub/grubenv",
        "/etc/conf.d/watchdog",
    )
    for name in files:
        shutil.copy2(name, backup / Path(name).name)
    fallback = json.loads((WS / "before.json").read_text())["fallback_hashes"]
    assert all(digest(Path(name)) == sha for name, sha in fallback.items())
    subprocess.run(["tar", "-xzf", str(WS / "kernel.tar.gz"), "-C", "/"], check=True)
    shutil.copy2(WS / "initramfs-r3", KWS / "artifacts" / ("initramfs-" + KREL))
    for path, expected in {
        KWS
        / "artifacts"
        / (
            "vmlinuz-" + KREL
        ): "4531223f35bd9f313a242c2aa5d7bece0b4ffd4161d28b41271914a9b54d4558",
        KWS
        / "artifacts"
        / (
            "initramfs-" + KREL
        ): "0e8444c328d070e0f6d422514116e19ccb13feb9e44d0bb4b289f0d1ea3eeb11",
        Path("/lib/modules")
        / KREL
        / "updates/amdgpu.ko.gz": "6ebd8393a9c7d25cd810fd89d4ef5defa187980cae64da0d37ee0d975715e8d1",
    }.items():
        assert digest(path) == expected, str(path)
    subprocess.run(["depmod", KREL], check=True)
    # Persist the unified hierarchy: an earlier manual mount is lost at reboot.
    subprocess.run(["rc-service", "cgroups", "start"], check=True)
    subprocess.run(["rc-update", "add", "cgroups", "boot"], check=True)
    (KWS / "logs").mkdir()
    service = Path("/etc/init.d/bc250-r3-bootguard")
    assert not service.exists()
    shutil.copy2(KWS / "helpers/bc250-r3-bootguard.openrc", service)
    service.chmod(0o755)
    subprocess.run(["rc-update", "add", service.name, "boot"], check=True)
    Path("/etc/conf.d/watchdog").write_text(
        'WATCHDOG_OPTS="-t 10 -T 120"\nWATCHDOG_DEV="/dev/watchdog0"\n'
    )
    subprocess.run(["rc-update", "add", "watchdog", "boot"], check=True)
    subprocess.run(["rc-service", "watchdog", "start"], check=True)
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    root = next(
        x
        for x in Path("/proc/cmdline").read_text().split()
        if x.startswith("root=UUID=")
    )
    uuid = root.split("=", 2)[2]
    assert uuid == "38603c88-3cb1-42d2-b4c6-f633cc5774c6"
    entry = "bc250-rocm-r3-7.2.10-trial"
    custom = Path("/etc/grub.d/40_custom")
    assert entry not in custom.read_text()
    parameters = f"{root} ro modules=sd-mod,usb-storage,ext4,nvme quiet rootfstype=ext4 amdgpu.sg_display=0 mitigations=off amdgpu.gttsize=15200 ttm.pages_limit=3891200 ttm.page_pool_size=3891200 amdgpu.gpu_recovery=0 amdgpu.bc250_cc_write_mode=3 amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=3 modprobe.blacklist=amdgpu panic=60 oops=panic softlockup_panic=1 nmi_watchdog=1 hardlockup_panic=1 sp5100_tco.heartbeat=180 watchdog.open_timeout=180 firmware_class.path={KWS}/firmware-from-r2"
    custom.write_text(
        custom.read_text()
        + f"""\n# BEGIN BC250 R3 SSH TRIAL\nmenuentry 'BC250 Linux {KREL} SSH trial' --id {entry} {{\n    set fallback="bc250-rocm-r2-kernel"\n    insmod gzio\n    insmod part_gpt\n    insmod ext2\n    search --no-floppy --fs-uuid --set=root {uuid}\n    linux {KWS}/artifacts/vmlinuz-{KREL} {parameters}\n    initrd {KWS}/artifacts/initramfs-{KREL}\n}}\n# END BC250 R3 SSH TRIAL\n"""
    )
    try:
        candidate = WS / "grub-new.cfg"
        subprocess.run(["grub-mkconfig", "-o", str(candidate)], check=True)
        subprocess.run(["grub-script-check", str(candidate)], check=True)
        assert 'set default="bc250-rocm-r2-kernel"' in candidate.read_text()
        shutil.copy2(candidate, "/boot/grub/grub.cfg.migration-new")
        Path("/boot/grub/grub.cfg.migration-new").replace("/boot/grub/grub.cfg")
    except Exception:
        shutil.copy2(backup / "40_custom", custom)
        shutil.copy2(backup / "grub.cfg", "/boot/grub/grub.cfg")
        raise
    assert all(digest(Path(name)) == sha for name, sha in fallback.items())
    result = {
        "status": "installed_not_selected",
        "fallback_hashes": fallback,
        "persistent_default": "bc250-rocm-r2-kernel",
        "entry": entry,
        "boot_arguments": parameters,
        "backup": str(backup),
        "limits": "Target trial/recovery untested; early/pre-watchdog hangs cannot be guaranteed recoverable",
    }
    (WS / "kernel-install.json").write_text(json.dumps(result, indent=2) + "\n")
    os.sync()
    print("MIGRATED_KERNEL_INSTALLED_NOT_SELECTED")


if __name__ == "__main__":
    main()
