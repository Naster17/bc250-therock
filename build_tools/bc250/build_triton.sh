#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail
source_dir="${1:?usage: build_triton.sh <source-directory> <working-directory> <dependency-directory> <therock-source>}"
work_dir="${2:?missing working directory}"
dependency_dir="${3:?missing dependency directory}"
therock_source="${4:?missing TheRock source directory}"
test "$(cat "$source_dir/GIT_REVISION")" = ee65ff3b85f99d52e678b252a99c8425fa46e329
mkdir -p "$work_dir"
cp -a "$source_dir/." "$work_dir/"

export LLVM_SYSPATH="$dependency_dir/llvm/llvm-5f07f818-ubuntu-x64-1"
export JSON_SYSPATH="$dependency_dir/json"
export TRITON_OFFLINE_BUILD=ON
export TRITON_BUILD_PROTON=OFF
export TRITON_BUILD_WITH_CCACHE=ON
export TRITON_PARALLEL_LINK_JOBS=1
export TRITON_BUILD_DIR="$work_dir/build"
export MAX_JOBS="${BC250_TRITON_JOBS:-4}"
helper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=$helper_dir/triton_pre_hook.cmake -DCMAKE_JOB_POOL_COMPILE=compile -DTRITON_BUILD_UT=OFF"
export CC=gcc CXX=g++
export CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}"
cache_dir="${BC250_CCACHE_DIR:-/workspace/cache}"
eval "$(python3 "$therock_source/build_tools/setup_ccache.py" \
    --dir "$cache_dir/triton-config" --local-path "$cache_dir/data" --no-reset-stats)"
cd "$work_dir"
make dev-install-triton
# Rebuild through the repository's required entry point before compiler tests.
make
build/bin/triton-opt --version
ccache --show-stats
