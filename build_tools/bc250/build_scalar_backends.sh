#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail
source_dir="${1:?source directory}"
build_dir="${2:?build directory}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BC250_BUILD_JOBS:-16}"
export CCACHE_MAXSIZE=20G
export OMP_NUM_THREADS=4
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DTHEROCK_GFX1013_SCALAR_BACKENDS=ON \
    -DTHEROCK_ENABLE_SPARSELT=ON \
    -DTHEROCK_ENABLE_HIPTENSOR=ON -DTHEROCK_HIPTENSOR_GENERIC_ONLY=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
cmake --build "$build_dir" --target hipBLASLt hipSPARSELt
CMAKE_BUILD_PARALLEL_LEVEL="${BC250_HIPTENSOR_JOBS:-12}" cmake --build "$build_dir" --target hipTensor
cmake --build "$build_dir" --target dist
ccache --show-stats
