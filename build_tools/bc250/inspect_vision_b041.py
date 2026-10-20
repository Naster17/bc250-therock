#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only extension ABI/dependency inspection in the retained containers."""

import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import ROCM_HOME

result = dict(
    python=sys.version,
    torch=str(torch.__version__),
    torch_file=torch.__file__,
    hip=torch.version.hip,
    cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    rocm_home=str(ROCM_HOME),
    torch_config=torch.__config__.show(),
    packages={
        name: bool(importlib.util.find_spec(name))
        for name in ("torchvision", "PIL", "numpy", "setuptools", "wheel", "ninja")
    },
    codec_libraries={},
)
for root in (
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/workspace/build/dist/rocm/lib"),
    Path("/workspace/build/dist/rocm/lib/rocm_sysdeps"),
):
    result["codec_libraries"][str(root)] = sorted(
        str(p)
        for pattern in ("libjpeg*", "libpng*", "libwebp*", "libz.so*")
        for p in root.glob(pattern)
    )
print(json.dumps(result, indent=2))
