#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
phase=python
prefix=/workspace/build/dist/rocm
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export ROCM_PATH="$prefix" HIP_PATH="$prefix" HIP_CLANG_PATH="$prefix/lib/llvm/bin"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export TMPDIR=/workspace/task/tmp
mkdir -p "$TMPDIR"
export CCACHE_MAXSIZE=2G
cache=/workspace/task/parser-cache-data
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/task/cache-config-$phase --local-path "$cache" --no-reset-stats)"
ccache --show-stats > /workspace/logs/ccache-before.txt
common=(-GNinja -DCMAKE_C_COMPILER="$prefix/lib/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$prefix/lib/llvm/bin/clang++"
 -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
 -DCMAKE_CXX_STANDARD=17 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -g0"
 -DCMAKE_JOB_POOLS=link=1 -DCMAKE_JOB_POOL_LINK=link
 -DCMAKE_INSTALL_PREFIX=/workspace/task/dependencies)
cmake -S /workspace/pybind -B /workspace/task/build-pybind "${common[@]}" \
 -DCMAKE_INSTALL_PREFIX=/workspace/task/pybind-dependencies -DPYBIND11_TEST=OFF > /workspace/logs/pybind-configure.log 2>&1
cmake --install /workspace/task/build-pybind > /workspace/logs/pybind-install.log 2>&1
cmake -S /workspace/task -B /workspace/task/build-r1 -GNinja \
  -DCMAKE_C_COMPILER="$prefix/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$prefix/lib/llvm/bin/clang++" \
  -DCMAKE_CXX_FLAGS="--rocm-path=$prefix" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -g0" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_JOB_POOLS=link=1 -DCMAKE_JOB_POOL_LINK=link \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/workspace/task/install-r1 \
  -DROCmCMakeBuildTools_DIR=/workspace/deps/share/rocmcmakebuildtools/cmake \
  -DCMAKE_PREFIX_PATH="/workspace/task/pybind-dependencies;/workspace/task/dependencies;$prefix;/workspace/build/third-party/nlohmann-json/dist;/workspace/build/third-party/msgpack-cxx/dist;/workspace/build/third-party/sysdeps/linux/sqlite3/build/dist/lib/rocm_sysdeps" \
  -DPYTHON_DISABLE_VERSIONS="3.6;3.7;3.8;3.9;3.10;3.11;3.13;3.14" -DGPU_TARGETS=gfx1013 -DBUILD_TESTING=OFF -DMIGRAPHX_ENABLE_GPU=ON \
  -DMIGRAPHX_ENABLE_PYTHON=ON -DMIGRAPHX_ENABLE_ONNX=ON -DMIGRAPHX_ENABLE_TENSORFLOW=OFF \
  -DMIGRAPHX_USE_EIGEN=OFF -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF -DMIGRAPHX_ENABLE_MLIR=OFF \
  -DMIGRAPHX_USE_AMDMLSS=OFF -DMIGRAPHX_USE_HIPBLASLT=OFF \
  -DMIGRAPHX_USE_ROCBLAS=ON -DMIGRAPHX_USE_MIOPEN=ON > /workspace/logs/configure.log 2>&1
cmake --build /workspace/task/build-r1 --target migraphx_py probe_migraphx_b064 probe_migraphx_b065 migraphx-hiprtc-driver --parallel 4 > /workspace/logs/build.log 2>&1
ccache --show-stats > /workspace/logs/ccache-after.txt
