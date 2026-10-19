#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Back up and update only the shared qualification helper under the runtime lock."""

import os
import shutil

from deploy_qwen35_training_ops import atomic_copy
from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS

if __name__ == "__main__":
    assert os.geteuid() == 0
    current = WS / "helpers/validate_training_n37.py"
    candidate = WS / "helpers/validate_training_n37.n39-next.py"
    assert (
        sha(current)
        == "31577b234e8a2826a1be2ba4de8fd3dc2206980bfef6eb08e8228c20926f8460"
    )
    assert (
        sha(candidate)
        == "f44aac7b349e93762bcd45e5c08fb3359bcd36a4a67cd3bf1866594ad09be601"
    )
    backup = WS / "recovery/training-n39-before"
    backup.mkdir()
    shutil.copy2(current, backup / current.name)
    atomic_copy(candidate, current, current.stat())
