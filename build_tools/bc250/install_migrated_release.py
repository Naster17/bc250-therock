#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage/activate an exact .104 userspace copy on the explicitly authorized .105."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from migrate_reference_host import inventory

WS = Path("/home/nik/bc250work/migration-105-20261023")
RELEASE = Path("/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "activate"))
    args = parser.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    if args.action == "stage":
        assert not RELEASE.exists()
        assert shutil.disk_usage("/").free > 20 * 1024**3
        RELEASE.mkdir(parents=True)
        for archive in ("runtime.tar.gz", "consumers-launchers.tar.gz"):
            subprocess.run(
                ["tar", "-xzf", str(WS / archive), "-C", str(RELEASE)], check=True
            )
        expected = json.loads((WS / "source-payload-files-locked.json").read_text())
        actual = {name: inventory(RELEASE / name) for name in ("rocm", "torch")}
        assert actual == expected, "Copied files differ from the source snapshot"
        (WS / "payload-verification.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "files_and_symlinks": {
                        name: len(value) for name, value in actual.items()
                    },
                    "source": "SHA256 inventory on .104 under exclusive runtime lock",
                },
                indent=2,
            )
            + "\n"
        )
        subprocess.run(
            [
                "python3",
                str(WS / "migrate_reference_host.py"),
                "seal",
                "--release",
                str(RELEASE),
                "--output",
                str(WS / "seal.json"),
            ],
            check=True,
        )
        runtime = json.loads((RELEASE / "runtime.json").read_text())
        subprocess.run(
            [
                "podman",
                "tag",
                runtime["image"],
                "localhost/bc250-stack:10.2-gfx1013-20261023-host105",
            ],
            check=True,
        )
        subprocess.run(["chown", "-hR", "0:0", str(RELEASE)], check=True)
        subprocess.run(["chmod", "-R", "go-w", str(RELEASE)], check=True)
        print("MIGRATED_RELEASE_STAGED_NOT_SELECTED")
    else:
        assert (
            json.loads((WS / "qualification-r3/final.json").read_text())["status"]
            == "passed"
        )
        current = Path("/opt/bc250-stack/current")
        assert not current.exists() and not current.is_symlink()
        manifest = json.loads((RELEASE / "runtime.json").read_text())
        manifest.update(
            migration_status="target bounded qualification passed",
            target_evidence=str(WS / "qualification-r3/final.json"),
        )
        (RELEASE / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name in ("bc250-torch", "bc250-train"):
            target = Path("/usr/local/bin") / name
            assert not target.exists(), "Preserve pre-existing launcher"
            shutil.copy2(RELEASE / name, target)
            target.chmod(0o755)
        temporary = current.with_name("current.migration-new")
        temporary.symlink_to(RELEASE)
        temporary.replace(current)
        print("MIGRATED_RELEASE_ACTIVATED")
    os.sync()


if __name__ == "__main__":
    main()
