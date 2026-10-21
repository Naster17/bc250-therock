#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve and extend the shared gate controller under the exclusive lock."""

import os
import shutil
from qualify_training_n33_on_host import WS

assert os.geteuid() == 0
backup = WS / "recovery/image-b045-before/qualify_training_n33_on_host.py"
assert backup.parent.is_dir() and not backup.exists()
source = WS / "helpers/qualify_training_n33_on_host.py"
shutil.copy2(source, backup)
os.replace(WS / "helpers/qualify_training_n33.image-next.py", source)
