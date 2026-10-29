#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
phase="$1"
prefix=/workspace/build/dist/rocm
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export HIP_PATH="$prefix" ROCM_PATH="$prefix" TMPDIR=/task/tmp CCACHE_MAXSIZE=2G
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
cache_settings="$(python3 /workspace/source/build_tools/setup_ccache.py --dir /task/cache-config --local-path /task/cache --no-reset-stats)"
eval "$cache_settings"
ccache --show-stats > /logs/ccache-before.txt
if [[ ! -e /task/venv/bin/python ]]; then
  python -m venv --system-site-packages /task/venv
fi
source /task/venv/bin/activate
if [[ "$phase" == configure ]]; then
  cmake -S /source/cmake -B /task/build -GNinja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$prefix/lib/llvm/bin/clang++" \
    -DCMAKE_C_COMPILER="$prefix/lib/llvm/bin/clang" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_SCAN_FOR_MODULES=OFF -DCMAKE_JOB_POOLS=link_pool=1 -DCMAKE_JOB_POOL_LINK=link_pool \
    -DCMAKE_PREFIX_PATH="/sdk;$prefix" -DAMD_MIGRAPHX_HOME=/sdk \
    -Donnxruntime_CMAKE_DEPS_MIRROR_DIR=/mirror -DFETCHCONTENT_FULLY_DISCONNECTED=OFF -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
    -Donnxruntime_USE_MIGRAPHX=ON -Donnxruntime_ENABLE_PYTHON=ON -Donnxruntime_BUILD_SHARED_LIB=ON \
    -Donnxruntime_BUILD_UNIT_TESTS=OFF -Donnxruntime_BUILD_BENCHMARKS=OFF \
    -Donnxruntime_ENABLE_TRAINING=OFF -Donnxruntime_ENABLE_TRAINING_OPS=OFF \
    -Donnxruntime_USE_CUDA=OFF -Donnxruntime_USE_TENSORRT=OFF -Donnxruntime_USE_DNNL=OFF \
    -Donnxruntime_USE_XNNPACK=OFF -Donnxruntime_USE_MIMALLOC=OFF -Donnxruntime_USE_VCPKG=OFF \
    -Donnxruntime_USE_1DS_TELEMETRY=OFF -Donnxruntime_USE_OPENMP=OFF -Donnxruntime_ENABLE_DLPACK=OFF \
    -DPython_EXECUTABLE=/task/venv/bin/python > /logs/configure.log 2>&1
else
  cmake --build /task/build --target onnxruntime onnxruntime_pybind11_state onnxruntime_providers_migraphx --parallel 4 > /logs/build.log 2>&1
fi
ccache --show-stats > /logs/ccache-after.txt
