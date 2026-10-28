#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build only the pinned ORT/SDK admission probe; no ORT package is installed."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from bc250_migraphx import registered
from bc250_torch import LaunchOptions, runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    preflight("onnxruntime-b069-build-" + args.attempt)
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 8 * 1024**3
    dest = WS / ("tmp/onnxruntime-b069/probe-" + args.attempt)
    dest.mkdir(parents=True)
    for source, target in [
        ("onnxruntime_b069_probe.cpp", "probe.cpp"),
        ("onnxruntime_b069_CMakeLists.txt", "CMakeLists.txt"),
    ]:
        shutil.copyfile(WS / "helpers" / source, dest / target)
    overlay = dest / "include/migraphx"
    overlay.mkdir(parents=True)
    shutil.copyfile(WS / "helpers/migraphx-b069-cpp20.hpp", overlay / "migraphx.hpp")
    shutil.copyfile(
        WS / "helpers/migraphx-b069-cpp20-test.cpp", dest / "header_test.cpp"
    )
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        record, sdk = registered(LaunchOptions())
        header = (sdk / "include/migraphx/migraphx.h").read_text()
        symbols = sorted(
            set(re.findall(r"migraphx_status\s+(migraphx_[a-zA-Z0-9_]+)\s*\(", header))
        )
        assert len(symbols) > 100
        # Preserve references in a separate externally visible translation unit.
        code = "#include <migraphx/migraphx.h>\nusing symbol = void (*)();\nsymbol api_symbols[] = {\n"
        code += "".join(
            " reinterpret_cast<symbol>(&" + name + "),\n" for name in symbols
        )
        code += "};\n"
        (dest / "api_symbols.cpp").write_text(code)
        atomic_json(
            dest / "api-symbols.json",
            dict(symbols=symbols, count=len(symbols), execution=False),
        )
        command = [
            "podman",
            "run",
            "--rm",
            "--name",
            "bc250-onnxruntime-b069-probe-" + args.attempt,
            "--network=none",
            "--read-only",
            "--cpus=4",
            "--memory=10g",
            "--memory-swap=12g",
            "-v",
            str(WS / "build") + ":/workspace/build:ro",
            "-v",
            str(WS / "source/build_tools") + ":/workspace/source/build_tools:ro",
            "-v",
            str(sdk) + ":/sdk:ro",
            "-v",
            str(dest) + ":/probe:rw",
            "fa3e8c254b6a",
            "bash",
            "-c",
            """
set -euo pipefail
prefix=/workspace/build/dist/rocm
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export TMPDIR=/probe/tmp CCACHE_MAXSIZE=2G
mkdir -p "$TMPDIR"
cache_settings="$(python3 /workspace/source/build_tools/setup_ccache.py --dir /probe/cache-config --local-path /probe/cache --no-reset-stats)"
eval "$cache_settings"
ccache --show-stats > /probe/ccache-before.txt
cmake --version > /probe/cmake-version.txt
cmake -S /probe -B /probe/build -GNinja -DCMAKE_CXX_COMPILER="$prefix/lib/llvm/bin/clang++" -DCMAKE_C_COMPILER="$prefix/lib/llvm/bin/clang" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_SCAN_FOR_MODULES=OFF -DCMAKE_PREFIX_PATH="/sdk;$prefix" > /probe/configure.log 2>&1
cmake --build /probe/build --parallel 4 > /probe/build.log 2>&1
/probe/build/header_cpp17 > /probe/header-cpp17.log
/probe/build/header_cpp20 > /probe/header-cpp20.log
ccache --show-stats > /probe/ccache-after.txt
""",
        ]
        with (dest / "controller.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=600
            )
        report = dict(
            status="built" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            command=command,
            c_api_symbols_requested=len(symbols),
            c_api_symbols_linked=len(symbols) if result.returncode == 0 else 0,
            registration=record,
            source_sha256={
                n: sha(dest / n)
                for n in (
                    "probe.cpp",
                    "api_symbols.cpp",
                    "CMakeLists.txt",
                    "include/migraphx/migraphx.hpp",
                    "header_test.cpp",
                )
            },
            runtime_changed=False,
            ort_provider_built=False,
        )
        atomic_json(dest / "final.json", report)
    assert result.returncode == 0, report
    print("ONNXRUNTIME_B069_PROBE_BUILD_PASS", len(symbols), flush=True)


if __name__ == "__main__":
    main()
