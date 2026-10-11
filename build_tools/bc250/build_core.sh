#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

source_dir="${1:?usage: build_core.sh <source-directory> <build-directory>}"
build_dir="${2:?usage: build_core.sh <source-directory> <build-directory>}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
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
    -DFLANG_PARALLEL_COMPILE_JOBS=1
cmake --build "$build_dir" --target dist
