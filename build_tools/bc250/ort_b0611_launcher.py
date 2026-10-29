#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run explicit packaged ORT/MIGraphX inference in the qualified CPython image."""
import os
from pathlib import Path
import sys


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "This package requires the qualified CPython 3.12 runtime image"
        )
    root = Path(__file__).resolve().parent.parent
    if not sys.argv[1:]:
        raise SystemExit("Supply Python arguments or a script")
    env = dict(os.environ)
    env["BC250_ORT_PACKAGE"] = str(root)
    env["PYTHONPATH"] = str(root / "python") + ":" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(root / "python/onnxruntime/capi")
        + ":"
        + str(root / "lib")
        + ":"
        + env.get("LD_LIBRARY_PATH", "")
    )
    # Match the admitted FP32 conversion/tuning policy on every fresh child.
    for name in (
        "FP16_ENABLE",
        "BF16_ENABLE",
        "FP8_ENABLE",
        "INT8_ENABLE",
        "EXHAUSTIVE_TUNE",
    ):
        env["ORT_MIGRAPHX_" + name] = "0"
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
