#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# A09 candidate uses the pinned Kineto ROCm backend and existing object cache.
set -euo pipefail
prefix=/workspace/build/dist/rocm
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/torch-config --local-path /workspace/cache/data --no-reset-stats)"
export ROCM_PATH="$prefix" HIP_PATH="$prefix" HIP_CLANG_PATH="$prefix/lib/llvm/bin"
export CMAKE_PREFIX_PATH="$prefix" PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export CC=gcc CXX=g++ CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_HIP_COMPILER_LAUNCHER=ccache
export CMAKE_HIP_COMPILER="$prefix/lib/llvm/bin/clang++"
export CMAKE_TOOLCHAIN_FILE=/workspace/helpers/torch_toolchain.cmake
export CMAKE_BUILD_TYPE=Release MAX_JOBS=12 CMAKE_JOB_POOLS=link=1 CMAKE_JOB_POOL_LINK=link
export PYTORCH_ROCM_ARCH=gfx1013 PYTORCH_BUILD_VERSION=2.13.0a0+gfx1013 PYTORCH_BUILD_NUMBER=1
export USE_ROCM=1 USE_CUDA=0 USE_CUDNN=0 USE_MAGMA=0 BLAS=OpenBLAS
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_ROCM_CK_GEMM=0 USE_ROCM_CK_SDPA=0 USE_MSLK=0
export USE_NCCL=0 USE_DISTRIBUTED=0 USE_KINETO=1 BUILD_TEST=0 USE_CUSPARSELT=0
export ROCM_SOURCE_DIR="$prefix"
export CMAKE_FRESH="${BC250_A09_FRESH:-1}"
cd /workspace/pytorch-work
test -f "$prefix/lib/cmake/hipsparselt/hipsparselt-config.cmake"
if [[ "$CMAKE_FRESH" == 1 ]]; then
    cache_snapshot="/workspace/logs/torch-a09-cache-${BC250_A09_LABEL:-initial}.txt"
    test ! -e "$cache_snapshot"
    cp build/CMakeCache.txt "$cache_snapshot"
fi
python -m pip install -e . -v --no-build-isolation
ccache --show-stats
