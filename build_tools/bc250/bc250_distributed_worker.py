#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate the native Torch, then preserve ordinary script argv/import semantics."""
from pathlib import Path
import runpy
import sys
from bc250_distributed_identity import check_runtime


def main() -> None:
    check_runtime()
    script = Path(sys.argv[1])
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
