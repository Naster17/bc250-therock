#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
prefix=/workspace/build/dist/rocm
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export ROCM_PATH="$prefix" HIP_PATH="$prefix" HIP_CLANG_PATH="$prefix/lib/llvm/bin"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export TMPDIR=/workspace/task/tmp
mkdir -p "$TMPDIR"
# Existing shared cache stays read-only. This durable task cache retains new entries.
export CCACHE_MAXSIZE=2G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/task/cache-config --local-path /workspace/task/cache-data --no-reset-stats)"
ccache --show-stats > /workspace/logs/ccache-before.txt
cmake -S /workspace/task -B /workspace/task/build-r1 -GNinja \
  -DCMAKE_C_COMPILER="$prefix/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$prefix/lib/llvm/bin/clang++" \
  -DCMAKE_CXX_FLAGS="--rocm-path=$prefix" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -g0" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_JOB_POOLS=link=1 -DCMAKE_JOB_POOL_LINK=link \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/workspace/task/install-r1 \
  -DROCmCMakeBuildTools_DIR=/workspace/deps/share/rocmcmakebuildtools/cmake \
  -DCMAKE_PREFIX_PATH="$prefix;/workspace/build/third-party/nlohmann-json/dist;/workspace/build/third-party/msgpack-cxx/dist;/workspace/build/third-party/sysdeps/linux/sqlite3/build/dist/lib/rocm_sysdeps" \
  -DGPU_TARGETS=gfx1013 -DBUILD_TESTING=OFF -DMIGRAPHX_ENABLE_GPU=ON \
  -DMIGRAPHX_ENABLE_PYTHON=OFF -DMIGRAPHX_ENABLE_ONNX=OFF -DMIGRAPHX_ENABLE_TENSORFLOW=OFF \
  -DMIGRAPHX_USE_EIGEN=OFF -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF -DMIGRAPHX_ENABLE_MLIR=OFF \
  -DMIGRAPHX_USE_AMDMLSS=OFF -DMIGRAPHX_USE_HIPBLASLT=OFF \
  -DMIGRAPHX_USE_ROCBLAS=ON -DMIGRAPHX_USE_MIOPEN=ON > /workspace/logs/configure.log 2>&1
cmake --build /workspace/task/build-r1 --target migraphx_gpu --parallel 4 > /workspace/logs/build.log 2>&1
ccache --show-stats > /workspace/logs/ccache-after.txt
