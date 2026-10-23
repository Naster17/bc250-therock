#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Make tested .105 r3 the normal boot with an r2 selection saved before entry."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from install_migrated_healthcheck import install as install_healthcheck
from migrate_reference_host import digest

WS = Path("/home/nik/bc250work/migration-105-20261023")
ENTRY = "bc250-host105-r3-release"


def main() -> None:
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    assert os.uname().release == "7.2.10-bc250rocm-r3"
    result = json.loads((WS / "qualification-r3/final.json").read_text())
    assert result["status"] == "passed" and result["checks"] == 656
    assert Path("/opt/bc250-stack/current").resolve() == Path(
        "/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105"
    )
    fallback = json.loads((WS / "before.json").read_text())["fallback_hashes"]
    assert all(digest(Path(path)) == sha for path, sha in fallback.items())
    backup = WS / "promotion-backup"
    backup.mkdir()
    for name in (
        "/etc/default/grub",
        "/etc/grub.d/40_custom",
        "/boot/grub/grub.cfg",
        "/boot/grub/grubenv",
    ):
        shutil.copy2(name, backup / Path(name).name)
    custom = Path("/etc/grub.d/40_custom")
    text = custom.read_text()
    assert ENTRY not in text
    block = text.split("# BEGIN BC250 R3 SSH TRIAL\n", 1)[1].split(
        "# END BC250 R3 SSH TRIAL", 1
    )[0]
    block = block.replace("bc250-rocm-r3-7.2.10-trial", ENTRY).replace(
        " SSH trial", " release"
    )
    # GRUB consumes one-shot next_entry at startup. Save a fresh fallback before
    # entering Linux, covering failure before the root acknowledgement service.
    block = block.replace(
        '    set fallback="bc250-rocm-r2-kernel"',
        '    set fallback="bc250-rocm-r2-kernel"\n    set next_entry="bc250-rocm-r2-kernel"\n    save_env next_entry',
    )
    block = block.replace(
        " firmware_class.path=", " bc250.host105=promoted firmware_class.path="
    )
    custom.write_text(
        text
        + "\n# BEGIN BC250 HOST105 RELEASE\n"
        + block
        + "# END BC250 HOST105 RELEASE\n"
    )
    defaults = Path("/etc/default/grub")
    assert "GRUB_DEFAULT=bc250-rocm-r2-kernel" in defaults.read_text()
    defaults.write_text(
        defaults.read_text().replace(
            "GRUB_DEFAULT=bc250-rocm-r2-kernel", "GRUB_DEFAULT=" + ENTRY
        )
    )
    guard = Path("/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py")
    shutil.copy2(guard, backup / "boot-guard-original.py")
    shutil.copy2(WS / "migration_boot_guard.py", guard)
    service = Path("/etc/init.d/bc250-host105-ready")
    assert not service.exists()
    service.write_text(
        f"""#!/sbin/openrc-run\nname="BC-250 host105 guarded release readiness"\ndescription="Retain saved r2 fallback until SSH, cooling and GPU references pass"\ndepend() {{\n    need sshd watchdog bc250-fan-curve cgroups\n}}\nstart() {{\n    ebegin "Checking promoted BC-250 release"\n    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {WS}/ready_migrated_kernel.py >> {WS}/promoted-console.log 2>&1\n    eend $?\n}}\n"""
    )
    service.chmod(0o755)
    subprocess.run(["rc-update", "add", service.name, "default"], check=True)
    install_healthcheck()
    try:
        candidate = WS / "grub-promoted.cfg"
        subprocess.run(["grub-mkconfig", "-o", str(candidate)], check=True)
        subprocess.run(["grub-script-check", str(candidate)], check=True)
        assert 'set default="' + ENTRY + '"' in candidate.read_text()
        shutil.copy2(candidate, "/boot/grub/grub.cfg.migration-new")
        Path("/boot/grub/grub.cfg.migration-new").replace("/boot/grub/grub.cfg")
    except Exception:
        shutil.copy2(backup / "40_custom", custom)
        shutil.copy2(backup / "grub", defaults)
        shutil.copy2(backup / "grub.cfg", "/boot/grub/grub.cfg")
        subprocess.run(["rc-update", "del", service.name, "default"], check=True)
        raise
    assert all(digest(Path(path)) == sha for path, sha in fallback.items())
    (WS / "promotion.json").write_text(
        json.dumps(
            {
                "status": "configured_not_boot_tested",
                "entry": ENTRY,
                "fallback": "r2 next_entry saved by GRUB before Linux; cleared after successful boot readiness",
                "backup": str(backup),
            },
            indent=2,
        )
        + "\n"
    )
    os.sync()
    print("MIGRATED_KERNEL_PROMOTED_CONFIGURATION_READY")


if __name__ == "__main__":
    main()
