#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run unchanged N2.3/N2.4/N2.5 gates against explicit device-step Schur."""

import argparse
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group", choices=("n23", "n24", "scale", "boundary", "failure"), required=True
    )
    args = parser.parse_args()
    import gfx1013_schur_candidate as candidate

    original = candidate.schur

    def device_schur(a, *, max_iterations=256):
        return original(
            a,
            max_iterations=max_iterations,
            qr_backend="device_control",
            q_backend="device",
        )

    old_hessenberg = candidate.hessenberg

    def device_hessenberg(a, **kwargs):
        kwargs["q_backend"] = "device"
        return old_hessenberg(a, **kwargs)

    candidate.hessenberg = device_hessenberg
    candidate.schur = device_schur
    script = (
        f"validate_solver_{args.group}.py"
        if args.group in ("n23", "n24")
        else "validate_solver_n25.py"
    )
    sys.argv = [script, "--output", str(args.output)]
    if args.group not in ("n23", "n24"):
        sys.argv += ["--group", args.group]
    runpy.run_path(str(Path(__file__).with_name(script)), run_name="__main__")


if __name__ == "__main__":
    main()
