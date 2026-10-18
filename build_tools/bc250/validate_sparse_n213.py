#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the source-owned N2.13 sparse triangular reference group."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--group",
        choices=("matrix", "edge", "extended", "unit", "trace"),
        default="matrix",
    )
    args = ap.parse_args()
    code = subprocess.call(
        [
            sys.executable,
            str(Path(__file__).with_name("test_bc250_sparse_triangular.py")),
            "--output",
            str(args.output),
            "--group",
            args.group,
        ]
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
