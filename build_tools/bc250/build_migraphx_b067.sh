#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
attempt="$2"
prefix=/workspace/build/dist/rocm
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export ROCM_PATH="$prefix" HIP_PATH="$prefix" HIP_CLANG_PATH="$prefix/lib/llvm/bin"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export TMPDIR=/workspace/task/tmp CCACHE_MAXSIZE=2G
mkdir -p "$TMPDIR"
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/task/cache-config-python --local-path /workspace/task/parser-cache-data --no-reset-stats)"
ccache --show-stats > /workspace/logs/ccache-before.txt
cmake --build /workspace/task/build-r1 --target migraphx_c driver migraphx_py migraphx-hiprtc-driver --parallel 4 > /workspace/logs/build.log 2>&1
cmake --install /workspace/task/build-r1 --prefix "/workspace/task/sdk-$attempt" > /workspace/logs/install.log 2>&1
ccache --show-stats > /workspace/logs/ccache-after.txt
