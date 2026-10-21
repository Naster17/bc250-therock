#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Back up shared workers before adding explicit read-only HIP candidate mounts."""

import os
import shutil
from qualify_training_n33_on_host import WS
from deploy_torch_a04_on_host import sha

assert os.geteuid() == 0
root = WS / "recovery/audio-b042-before"
for source, staged in (
    ("validate_torch_a04_on_host.py", "validate_torch_a04.audio-next.py"),
    ("qualify_training_n33_on_host.py", "qualify_training_n33.audio-runtime-next.py"),
):
    backup = root / (source + ".before-runtime")
    assert not backup.exists()
    shutil.copy2(WS / "helpers" / source, backup)
    os.replace(WS / "helpers" / staged, WS / "helpers" / source)
    print(source, sha(backup), sha(WS / "helpers" / source))
