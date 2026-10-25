#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run Python with this opt-in gfx1013 SDK inside its qualified ROCm image."""
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "share/bc250-sdk/manifest.json").read_text())
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("This SDK requires the qualified CPython 3.12 runtime image")
    if sys.argv[1:] == ["--sdk-info"]:
        print(json.dumps(manifest, indent=2))
        return
    if len(sys.argv) < 2:
        raise SystemExit("Usage: bc250-migraphx.py SCRIPT.py [arguments] | --sdk-info")
    env = dict(os.environ)
    env["BC250_MIGRAPHX_SDK"] = str(root)
    env["PYTHONPATH"] = str(root / "lib") + ":" + env.get("PYTHONPATH", "")
    rocm = Path(env.get("ROCM_PATH", "/workspace/build/dist/rocm"))
    env["LD_LIBRARY_PATH"] = ":".join(
        (
            str(root / "lib"),
            str(root / "lib/migraphx/lib"),
            str(rocm / "lib/rocm_sysdeps/lib"),
            env.get("LD_LIBRARY_PATH", ""),
        )
    )
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
