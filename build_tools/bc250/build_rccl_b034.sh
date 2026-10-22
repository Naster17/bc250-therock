#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
prefix=/workspace/build/dist/rocm
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/rccl-b034-config --local-path /workspace/cache/data --no-reset-stats)"
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export HIP_PATH="$prefix" ROCM_PATH="$prefix" HIP_CLANG_PATH="$prefix/lib/llvm/bin"
cmake -S /workspace/rccl/source/projects/rccl -B /workspace/rccl/build-r1 -GNinja \
  -DCMAKE_CXX_COMPILER="$prefix/lib/llvm/bin/clang++" \
  -DCMAKE_C_COMPILER="$prefix/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_JOB_POOLS=link=1 -DCMAKE_JOB_POOL_LINK=link \
  -DCMAKE_PREFIX_PATH="$prefix" -DROCM_PATH="$prefix" \
  -DCMAKE_INSTALL_PREFIX=/workspace/rccl/install-r1 \
  -DEXPLICIT_ROCM_VERSION=10.2.0 -DGPU_TARGETS=gfx1013 \
  -DBUILD_TESTS=OFF -DINSTALL_DEPENDENCIES=OFF -DBUILD_NCCL4PY=OFF \
  -DBUILD_PROFILER_INSPECTOR=OFF -DBUILD_PLUGIN_EXAMPLES=OFF \
  -DGENERATE_SYM_KERNELS=OFF -DENABLE_ROCSHMEM=OFF -DENABLE_ROCSHMEM_GIN=OFF \
  -DENABLE_MSCCLPP=OFF -DENABLE_MSCCL_KERNEL=OFF -DROCTX=OFF \
  -DONLY_FUNCS="AllReduce RING/TREE LL/SIMPLE Sum/Prod/MinMax i8/u8/i32/u32/i64/u64/f16/f32/f64/bf16|Reduce RING LL/SIMPLE Sum i8/u8/i32/u32/i64/u64/f16/f32/f64/bf16|ReduceScatter RING LL/SIMPLE Sum i8/u8/i32/u32/i64/u64/f16/f32/f64/bf16|Broadcast RING LL/SIMPLE|AllGather RING LL/SIMPLE|SendRecv"
cmake --build /workspace/rccl/build-r1 --parallel 4
cmake --install /workspace/rccl/build-r1
ccache --show-stats
