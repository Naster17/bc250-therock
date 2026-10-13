#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

source_dir="${1:?usage: build_torch.sh <source-directory> <working-directory> <rocm-prefix> <therock-source>}"
work_dir="${2:?missing working directory}"
rocm_prefix="${3:?missing ROCm prefix}"
therock_source="${4:?missing TheRock source directory}"
expected_revision="${BC250_TORCH_SOURCE_REVISION:-b36877506d38eb74abcd01d667e9b05d6b60430a}"
test "$(cat "$source_dir/GIT_REVISION")" = "$expected_revision"
test -x "$rocm_prefix/bin/amdclang++"
test -f "$rocm_prefix/lib/libMIOpen.so"
test -f "$rocm_prefix/lib/librocblas.so"

mkdir -p "$work_dir"
cp -a "$source_dir/." "$work_dir/"
cache_dir="${BC250_CCACHE_DIR:-/workspace/cache}"
export CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}"
eval "$(python3 "$therock_source/build_tools/setup_ccache.py" \
    --dir "$cache_dir/torch-config" --local-path "$cache_dir/data" --no-reset-stats)"

export ROCM_PATH="$rocm_prefix"
export HIP_PATH="$rocm_prefix"
export HIP_CLANG_PATH="$rocm_prefix/lib/llvm/bin"
export CMAKE_PREFIX_PATH="$rocm_prefix"
export PATH="$rocm_prefix/bin:$rocm_prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$rocm_prefix/lib:$rocm_prefix/lib/llvm/lib"
export CC=gcc CXX=g++
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_HIP_COMPILER_LAUNCHER=ccache
export CMAKE_HIP_COMPILER="$rocm_prefix/lib/llvm/bin/clang++"
helper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CMAKE_TOOLCHAIN_FILE="$helper_dir/torch_toolchain.cmake"
export CMAKE_BUILD_TYPE=Release
export MAX_JOBS="${BC250_TORCH_JOBS:-12}"
export CMAKE_JOB_POOLS=link=1
export CMAKE_JOB_POOL_LINK=link
export PYTORCH_ROCM_ARCH=gfx1013
export PYTORCH_BUILD_VERSION=2.13.0a0+gfx1013
export PYTORCH_BUILD_NUMBER=1
export USE_ROCM=1 USE_CUDA=0
export USE_CUDNN=0 USE_MAGMA=0
export BLAS=OpenBLAS
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0
export USE_ROCM_CK_GEMM=0 USE_ROCM_CK_SDPA=0 USE_MSLK=0
export USE_NCCL=0 USE_DISTRIBUTED=0 USE_KINETO=0
export BUILD_TEST=0 USE_CUSPARSELT=0

cd "$work_dir"
# Regenerate configuration after a failed configure, preserving compiled objects.
rm -f build/CMakeCache.txt
python -m pip install -e . -v --no-build-isolation
python -m pip freeze
ccache --show-stats
