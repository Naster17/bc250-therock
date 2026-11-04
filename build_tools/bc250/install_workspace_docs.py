#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install a verified documentation snapshot with a stable host entry point."""
import argparse
import hashlib
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    assert a.name and "/" not in a.name and a.name not in [".", ".."]
    assert hashlib.sha256(a.archive.read_bytes()).hexdigest() == a.sha256
    base = Path("/home/nik/bc250work/docs")
    dest = base / "snapshots" / a.name
    dest.mkdir(parents=True)
    with tarfile.open(a.archive) as t:
        t.extractall(dest, filter="data")
    (dest / "flow.md").symlink_to("therock/flow.md")
    (dest / "docs/development").symlink_to(
        "../therock/docs/development", target_is_directory=True
    )
    readme = dest / "README.md"
    readme.write_text(
        readme.read_text()
        .replace("(../", "(")
        .replace("(WORKSPACE_LAYOUT.md)", "(docs/WORKSPACE_LAYOUT.md)")
    )
    temp = base / "current-new"
    temp.symlink_to(Path("snapshots") / a.name)
    temp.replace(base / "current")
    (base / "README.md").write_text(
        """# BC-250 host documentation\n\nStart with [the current documentation index](current/README.md).\n\nAgents: read [canonical progress snapshot](current/flow.md) and\n[workspace instructions](current/AGENTS.md) before changing anything.\nThe authoritative source remains the workstation workspace; this is a dated copy.\nNative runtime manifests take precedence over historical reports.\n"""
    )
    print(dest)


if __name__ == "__main__":
    main()
