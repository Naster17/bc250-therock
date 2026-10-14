#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail
work_dir="${1:?Torch working directory}"
rocm_prefix="${2:?ROCm prefix}"
source_dir="${3:?TheRock source directory}"
helpers="${4:?helpers directory}"
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" --dir /workspace/cache/torch-config --local-path /workspace/cache/data --no-reset-stats)"
export CCACHE_MAXSIZE=20G
export ROCM_PATH="$rocm_prefix" HIP_PATH="$rocm_prefix"
export HIP_CLANG_PATH="$rocm_prefix/lib/llvm/bin"
export CMAKE_PREFIX_PATH="$rocm_prefix"
export PATH="$rocm_prefix/bin:$rocm_prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$rocm_prefix/lib:$rocm_prefix/lib/llvm/lib"
export CC=gcc CXX=g++ CMAKE_BUILD_TYPE=Release
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_HIP_COMPILER_LAUNCHER=ccache
export CMAKE_HIP_COMPILER="$rocm_prefix/lib/llvm/bin/clang++"
export CMAKE_TOOLCHAIN_FILE="$helpers/torch_toolchain.cmake"
export MAX_JOBS="${BC250_TORCH_JOBS:-12}" CMAKE_JOB_POOLS=link=1 CMAKE_JOB_POOL_LINK=link
export PYTORCH_ROCM_ARCH=gfx1013 PYTORCH_BUILD_VERSION=2.13.0a0+gfx1013 PYTORCH_BUILD_NUMBER=1
export USE_ROCM=1 USE_CUDA=0 USE_CUDNN=0 USE_MAGMA=0 BLAS=OpenBLAS
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0
export USE_ROCM_CK_GEMM=0 USE_ROCM_CK_SDPA=0 USE_MSLK=0
export USE_NCCL=0 USE_DISTRIBUTED=0 USE_KINETO=0 BUILD_TEST=0 USE_CUSPARSELT=0
cd "$work_dir"
# A cached CMake configure does not regenerate HIP sources after CUDA edits.
# Hipify compares contents and preserves unchanged generated files and objects.
python tools/amd_build/build_amd.py --out-of-place-only
# Required editable-pip workflow: preserves the existing objects and cache.
python -m pip install -e . -v --no-build-isolation
