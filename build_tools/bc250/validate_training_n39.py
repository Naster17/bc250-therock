#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reuse the qualified CLI contracts with explicitly owned disk-backed scratch."""

from validate_qwen35_training_cli import WS
from validate_training_n37 import main

if __name__ == "__main__":
    main(scratch_parent=WS / "experiments")
