#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail
source_dir="${1:?source directory}"
build_dir="${2:?build directory}"
prefix="$build_dir/dist/rocm"
ck="$source_dir/rocm-libraries/projects/composablekernel"
output="$build_dir/bc250-coverage/ck-tile-priority1"
mkdir -p "$output"
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
export CCACHE_MAXSIZE=20G
rapidjson="$source_dir/rocm-libraries/projects/hipblaslt/tensilelite/Source/third_party/rapidjson/include"
if [[ ! -f "$rapidjson/rapidjson/document.h" ]]; then
    rapidjson="$build_dir/third-party/rapidjson/stage/include"
fi
compile() {
    local name="$1"
    shift
    ccache "$prefix/bin/amdclang++" -x hip -O2 -std=c++20 --offload-arch=gfx1013 \
        -DCK_CMAKE_GPU_TARGET_IDS=0x1013 -DCK_TILE_USE_WMMA=0 \
        -I"$ck/include" -I"$ck/library/include" -I"$rapidjson" \
        -I"$ck/example/ck_tile/09_topk_softmax" \
        -I"$build_dir/math-libs/composable_kernel/build/include" \
        "$@" -o "$output/$name"
}
compile topk "$source_dir/build_tools/bc250/ck_tile_topk_smoke.cpp" \
    "$ck/example/ck_tile/09_topk_softmax/topk_softmax_api.cpp" &
p1=$!
compile pool3d "$ck/example/ck_tile/36_pooling/pool3d.cpp" &
p2=$!
compile layernorm "$source_dir/build_tools/bc250/ck_tile_layernorm_smoke.cpp" &
p3=$!
compile scalar_gemm "$source_dir/build_tools/bc250/ck_tile_scalar_gemm_smoke.cpp" &
p4=$!
status=0
for pid in "$p1" "$p2" "$p3" "$p4"; do wait "$pid" || status=1; done
exit "$status"
