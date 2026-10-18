#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retain the existing bounded floating Lt API and dispatch gates."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, marker, count in [
        ("hipblaslt_scalar_smoke", "HIPBLASLT_SCALAR_PASS", 403),
        ("hipblaslt_priority1_smoke", "HIPBLASLT_PRIORITY1_PASS", 561),
        ("hipblaslt_n18_smoke", "HIPBLASLT_SKINNY_PASS", 664),
    ]:
        p = subprocess.run(
            ["/home/nik/bc250work/rocm-10.2-gfx1013/build/bc250-coverage/" + name],
            capture_output=True,
            text=True,
            timeout=120,
        )
        (args.output.parent / (name + ".log")).write_text(p.stdout + p.stderr)
        assert p.returncode == 0, (name, p.stdout[-2000:], p.stderr[-2000:])
        found = re.search(marker + r" cases=(\d+)", p.stdout)
        assert found and int(found[1]) == count, (
            name,
            found.groups() if found else None,
        )
        rows.append(dict(status="passed", suite=name, cases=count))
        args.output.write_text(
            json.dumps(dict(status="running", results=rows), indent=2) + "\n"
        )
    args.output.write_text(
        json.dumps(dict(status="passed", results=rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
