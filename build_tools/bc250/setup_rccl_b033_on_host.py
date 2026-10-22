#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve the shared qualifier before extending its accepted task IDs."""

import os
import shutil
from qualify_training_n33_on_host import WS

assert os.geteuid() == 0
backup = WS / "recovery/rccl-b033-before"
backup.mkdir()
shutil.copy2(
    WS / "helpers/qualify_training_n33_on_host.py",
    backup / "qualify_training_n33_on_host.py",
)
os.replace(
    WS / "helpers/qualify_training_n33.rccl-next.py",
    WS / "helpers/qualify_training_n33_on_host.py",
)
