#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only loader diagnostic; no GPU initialization."""

import ctypes
import os
from pathlib import Path
import subprocess

os.environ["TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS"] = "1"
import torch
import torchvision

image = next(Path(torchvision.__file__).parent.glob("image*.so"))
print(subprocess.check_output(["ldd", str(image)], text=True), flush=True)
ctypes.CDLL(str(image))
print("IMAGE_LOADED", torch.__version__, flush=True)
