#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail
source_dir="${1:?source directory}"
build_dir="${2:?build directory}"
export CCACHE_MAXSIZE=20G CMAKE_BUILD_PARALLEL_LEVEL="${BC250_BUILD_JOBS:-16}"
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DTHEROCK_GFX1013_SCALAR_BACKENDS=ON \
    -DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
cmake --build "$build_dir" --target hipBLASLt
CMAKE_BUILD_PARALLEL_LEVEL="${BC250_DNN_JOBS:-6}" cmake --build "$build_dir" --target hipblasltprovider miopenprovider hipdnn_integration_tests
cmake --build "$build_dir" --target composable_kernel
if [[ "${BC250_STAGE_ONLY:-0}" != 1 ]]; then
    cmake --build "$build_dir" --target dist
fi
ccache --show-stats
