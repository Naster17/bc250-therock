#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail
source_dir="${1:?source directory}"
build_dir="${2:?build directory}"
prefix="$build_dir/dist/rocm"
ck="$source_dir/rocm-libraries/projects/composablekernel"
output="$build_dir/bc250-coverage/ck-tile"
mkdir -p "$output"
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
export CCACHE_MAXSIZE=20G
# These standalone generic examples do not require a full CK/matrix-codegen rebuild.
rapidjson="$source_dir/rocm-libraries/projects/hipblaslt/tensilelite/Source/third_party/rapidjson/include"
if [[ ! -f "$rapidjson/rapidjson/document.h" ]]; then
    rapidjson="$build_dir/third-party/rapidjson/stage/include"
fi
compile() {
    local relative="$1" name="$2"
    ccache "$prefix/bin/amdclang++" -x hip -O2 -std=c++20 --offload-arch=gfx1013 \
        -DCK_CMAKE_GPU_TARGET_IDS=0x1013 -DCK_TILE_USE_WMMA=0 \
        -I"$ck/include" -I"$ck/library/include" -I"$rapidjson" \
        -I"$build_dir/math-libs/composable_kernel/build/include" \
        "$ck/example/ck_tile/$relative" -o "$output/$name"
}
# Bound template compilation memory while still using the CPU concurrently.
compile 05_reduce/reduce.cpp reduce &
p1=$!
compile 06_permute/permute.cpp permute &
p2=$!
compile 21_elementwise/elementwise_example_transpose.cpp transpose &
p3=$!
compile 21_elementwise/elementwise_example_unary.cpp unary &
p4=$!
compile 36_pooling/pool3d.cpp pool3d &
p5=$!
compile 10_rmsnorm2d/example_rmsnorm2d_fwd.cpp rmsnorm &
p6=$!
status=0
for pid in "$p1" "$p2" "$p3" "$p4" "$p5" "$p6"; do
    wait "$pid" || status=1
done
exit "$status"
