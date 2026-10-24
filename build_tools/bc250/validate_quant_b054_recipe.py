#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the retained full recipe fixture against the isolated candidate module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "b054-candidate"))
from validate_quant_b051 import main

if __name__ == "__main__":
    main()
