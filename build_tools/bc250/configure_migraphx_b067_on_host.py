#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Configure/build an external stable-API SDK consumer without SDK source/build mounts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight
from package_migraphx_b067_on_host import STORAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--attempt", required=True, choices=("r1", "r2", "r3"))
    args = parser.parse_args()
    assert args.install.parent == STORAGE and args.install.is_dir()
    preflight("migraphx-b067-consumer")
    dest = STORAGE / ("consumer-" + args.attempt)
    dest.mkdir()
    shutil.copyfile(WS / "helpers/migraphx_b067_consumer.cpp", dest / "consumer.cpp")
    shutil.copyfile(
        WS / "helpers/migraphx_b067_consumer_CMakeLists.txt", dest / "CMakeLists.txt"
    )
    command = [
        "podman",
        "run",
        "--name",
        "bc250-migraphx-b067-consumer-" + args.attempt,
        "--network=none",
        "--read-only",
        "--cpus=4",
        "--memory=10g",
        "--memory-swap=12g",
        "-v",
        f'{WS / "build"}:/workspace/build:ro',
        "-v",
        f'{WS / "source"}:/workspace/source:ro',
        "-v",
        f"{args.install}:/sdk:ro",
        "-v",
        f"{dest}:/consumer:rw",
        "-v",
        f'{STORAGE / "parser-cache-data"}:/cache:rw',
        "fa3e8c254b6a",
        "bash",
        "-c",
        """
set -euo pipefail
prefix=/workspace/build/dist/rocm
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export TMPDIR=/consumer/tmp CCACHE_MAXSIZE=2G
mkdir -p "$TMPDIR"
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /consumer/cache-config --local-path /cache --no-reset-stats)"
cmake -S /consumer -B /consumer/build -GNinja -DCMAKE_CXX_COMPILER="$prefix/lib/llvm/bin/clang++" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/sdk;$prefix" > /consumer/configure.log 2>&1
cmake --build /consumer/build --parallel 4 > /consumer/build.log 2>&1
""",
    ]
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        with (dest / "controller.log").open("w") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, timeout=600
            )
        report = dict(
            status="built" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            command=command,
            sdk=str(args.install),
            runtime_changed=False,
        )
        atomic_json(dest / "final.json", report)
    assert result.returncode == 0, report


if __name__ == "__main__":
    main()
