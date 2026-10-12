#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail
work_dir="${1:?usage: validate_triton_lowering.sh <working-directory> <dependency-directory> <output-directory>}"
dependency_dir="${2:?missing dependency directory}"
output_dir="${3:?missing output directory}"
helper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LLVM_SYSPATH="$dependency_dir/llvm/llvm-5f07f818-ubuntu-x64-1"
export JSON_SYSPATH="$dependency_dir/json"
export TRITON_OFFLINE_BUILD=ON TRITON_BUILD_PROTON=OFF
export TRITON_BUILD_DIR="$work_dir/build"
export MAX_JOBS=4
export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=$helper_dir/triton_pre_hook.cmake -DCMAKE_JOB_POOL_COMPILE=compile -DTRITON_BUILD_UT=OFF"
cd "$work_dir"
make
python "$helper_dir/validate_triton_lowering.py" \
    --triton-opt "$work_dir/build/bin/triton-opt" --output-dir "$output_dir" \
    --llvm-bin "$LLVM_SYSPATH/bin"
