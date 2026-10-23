#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install .105 boot readiness under the release, independent of migration scratch."""

import json
import os
import shutil
from pathlib import Path

from migrate_reference_host import digest

WS = Path("/home/nik/bc250work/migration-105-20261023")
RELEASE = Path("/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105")
FILES = (
    "ready_migrated_kernel.py",
    "qualify_migrated_host.py",
    "migrate_reference_host.py",
    "bc250_torch.py",
    "validate_cleanup_runtime.py",
)


def install() -> None:
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    target = RELEASE / "healthchecks"
    assert not target.exists(), "Preserve an existing operational healthcheck package"
    target.mkdir()
    hashes = {}
    for name in FILES:
        shutil.copy2(WS / name, target / name)
        os.chown(target / name, 0, 0)
        (target / name).chmod(0o644)
        hashes[name] = digest(target / name)
    data = Path("/var/lib/bc250-stack/host105")
    data.mkdir(parents=True, exist_ok=True)
    assert data.stat().st_uid == 0
    service = Path("/etc/init.d/bc250-host105-ready")
    if service.exists():
        shutil.copy2(service, WS / "healthcheck-service-before")
        text = service.read_text().replace(
            str(WS / "ready_migrated_kernel.py"),
            str(target / "ready_migrated_kernel.py"),
        )
        text = text.replace(
            str(WS / "promoted-console.log"), "/var/log/bc250-host105-ready.log"
        )
        text = text.replace(
            "    /usr/bin/python3", "    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3"
        )
        service.write_text(text)
    evidence = RELEASE / "qualification"
    evidence.mkdir()
    shutil.copy2(WS / "qualification-r3/final.json", evidence / "host105-r3.json")
    manifest = json.loads((RELEASE / "runtime.json").read_text())
    manifest["target_evidence"] = str(evidence / "host105-r3.json")
    manifest["operational_boot_tools"] = {
        "path": str(target),
        "sha256": hashes,
        "data": str(data),
        "console": "/var/log/bc250-host105-ready.log",
    }
    (RELEASE / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (WS / "healthcheck-install.json").write_text(
        json.dumps(
            {
                "status": "installed_boot_test_pending",
                "path": str(target),
                "sha256": hashes,
                "data": str(data),
                "native_payload_changed": False,
            },
            indent=2,
        )
        + "\n"
    )
    os.sync()
    print("OPERATIONAL_HEALTHCHECK_PACKAGE_INSTALLED")


if __name__ == "__main__":
    install()
