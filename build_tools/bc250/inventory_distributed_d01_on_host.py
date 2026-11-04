#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Hash the selected immutable deployment inputs before constructing a matched update."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path("/opt/bc250-stack/current").resolve()
    data = {}
    for p in sorted((root / "rocm").resolve().rglob("*")):
        rel = str(p.relative_to((root / "rocm").resolve()))
        if p.is_symlink():
            data[rel] = dict(link=str(p.readlink()))
        elif p.is_file():
            with p.open("rb") as f:
                h = hashlib.file_digest(f, "sha256").hexdigest()
            data[rel] = dict(sha256=h, bytes=p.stat().st_size)
    result = dict(
        release=str(root),
        runtime=json.loads((root / "runtime.json").read_text()),
        rocm=data,
    )
    Path("/home/nik/bc250work/distributed-d01-rocm-inventory.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
