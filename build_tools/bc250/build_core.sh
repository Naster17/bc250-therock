#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

source_dir="${1:?usage: build_core.sh <source-directory> <build-directory>}"
build_dir="${2:?usage: build_core.sh <source-directory> <build-directory>}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
cache_dir="${BC250_CCACHE_DIR:-/workspace/cache}"
export CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}"
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" \
    --dir "$cache_dir" --local-path "$cache_dir/data" --no-reset-stats)"

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
    -DTHEROCK_BOOTSTRAP_CCACHE_MASQUERADE=ON \
    -DTHEROCK_AMDGPU_FAMILIES=gfx1013 \
    -DTHEROCK_TEST_AMDGPU_FAMILIES=gfx1013 \
    -DTHEROCK_ENABLE_ALL=OFF \
    -DTHEROCK_ENABLE_COMPILER=ON \
    -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
    -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
    -DTHEROCK_BUILD_TESTING=ON \
    -DTHEROCK_ENABLE_LLVM_TESTS=OFF \
    -DTHEROCK_BUILD_COMGR_TESTS=OFF \
    -DTHEROCK_BACKGROUND_BUILD_JOBS=1 \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DFLANG_PARALLEL_COMPILE_JOBS="${BC250_FLANG_JOBS:-4}"
cmake --build "$build_dir" --target dist
