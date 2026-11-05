#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refuse accidental overlay replacement of the pinned Torch before user code."""
import hashlib
import json
from pathlib import Path


def check_runtime() -> None:
    import torch
    import torch._C

    base = Path(__file__).resolve().parent
    pin = json.loads((base / "torch-identity.json").read_text())
    root = Path("/workspace/pytorch-work")
    if Path(torch.__file__).resolve() != root / "torch/__init__.py":
        raise RuntimeError(
            "Consumer overlay replaced the qualified Torch Python package"
        )
    if Path(torch._C.__file__).resolve() != root / pin["C_extension"]:
        raise RuntimeError("Consumer overlay replaced the qualified Torch C extension")
    for relative, digest in pin["sha256"].items():
        path = root / relative
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise RuntimeError(
                    f"Qualified distributed Torch identity mismatch: {relative}"
                )
    if torch.__version__ != pin["version"]:
        raise RuntimeError("Qualified distributed Torch version mismatch")
