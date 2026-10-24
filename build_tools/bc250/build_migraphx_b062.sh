#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
prefix=/workspace/build/dist/rocm
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/migraphx-b062 --local-path /workspace/cache/data --no-reset-stats)"
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export ROCM_PATH="$prefix" HIP_PATH="$prefix" HIP_CLANG_PATH="$prefix/lib/llvm/bin"
export TMPDIR=/workspace/task/tmp
mkdir -p "$TMPDIR"
ccache --show-stats > /workspace/logs/ccache-before.txt
for dpp in 1 0; do
    ccache "$prefix/bin/hipcc" --offload-arch=gfx1013 -std=c++17 -O2 -g0 \
      -DMIGRAPHX_NGLOBAL=128 -DMIGRAPHX_NLOCAL=128 -DMIGRAPHX_WAVEFRONTSIZE=32 \
      -DMIGRAPHX_HAS_DPP="$dpp" -I/workspace/task/source/src/targets/gpu/kernels/include \
      /workspace/helpers/probe_migraphx_b062.hip \
      -Wl,-rpath,"$prefix/lib" -o /workspace/task/tmp/probe-dpp"$dpp"
    ccache "$prefix/bin/hipcc" --offload-arch=gfx1013 --offload-device-only -c \
      -std=c++17 -O2 -g0 -DMIGRAPHX_NGLOBAL=128 -DMIGRAPHX_NLOCAL=128 \
      -DMIGRAPHX_WAVEFRONTSIZE=32 -DMIGRAPHX_HAS_DPP="$dpp" \
      -I/workspace/task/source/src/targets/gpu/kernels/include \
      /workspace/helpers/probe_migraphx_b062.hip -o /workspace/task/tmp/device-dpp"$dpp".o
    "$prefix/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o \
      --targets=hipv4-amdgcn-amd-amdhsa--gfx1013 \
      --input=/workspace/task/tmp/device-dpp"$dpp".o --output=/workspace/task/tmp/device-dpp"$dpp".elf
    "$prefix/lib/llvm/bin/llvm-objdump" -d /workspace/task/tmp/device-dpp"$dpp".elf > /workspace/logs/device-dpp"$dpp".asm
    "$prefix/lib/llvm/bin/llvm-readelf" --notes /workspace/task/tmp/device-dpp"$dpp".elf > /workspace/logs/device-dpp"$dpp"-notes.txt
done
ccache --show-stats > /workspace/logs/ccache-after.txt
set +e
cmake -S /workspace/task/source -B /workspace/task/tmp/configure -GNinja \
  -DCMAKE_C_COMPILER="$prefix/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$prefix/lib/llvm/bin/clang++" \
  -DCMAKE_CXX_FLAGS="--rocm-path=$prefix" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_JOB_POOLS=link=1 -DCMAKE_JOB_POOL_LINK=link \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/workspace/task/install \
  -DROCmCMakeBuildTools_DIR=/workspace/source/base/rocm-cmake/share/rocmcmakebuildtools/cmake \
  -DCMAKE_PREFIX_PATH="$prefix;/workspace/build/third-party/nlohmann-json/dist;/workspace/build/third-party/msgpack-cxx/dist;/workspace/build/third-party/sysdeps/linux/sqlite3/build/dist/lib/rocm_sysdeps" \
  -DGPU_TARGETS=gfx1013 -DBUILD_TESTING=OFF -DMIGRAPHX_ENABLE_GPU=ON \
  -DMIGRAPHX_ENABLE_PYTHON=OFF -DMIGRAPHX_ENABLE_ONNX=OFF -DMIGRAPHX_ENABLE_TENSORFLOW=OFF \
  -DMIGRAPHX_USE_EIGEN=OFF -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF -DMIGRAPHX_ENABLE_MLIR=OFF \
  -DMIGRAPHX_USE_AMDMLSS=OFF -DMIGRAPHX_USE_HIPBLASLT=OFF \
  -DMIGRAPHX_USE_ROCBLAS=ON -DMIGRAPHX_USE_MIOPEN=ON > /workspace/logs/configure.log 2>&1
configure_exit=$?
set -e
printf '%s\n' "$configure_exit" > /workspace/logs/configure-exit.txt
printf 'MIGRAPHX_B062_PROBES_BUILT configure_exit=%s\n' "$configure_exit"
