#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
prefix=/workspace/build/dist/rocm
stage=/workspace/build/math-libs/composable_kernel/stage
for mode in nodl dl; do
    flags=()
    if [[ "$mode" == dl ]]; then flags+=(-DDL_KERNELS); fi
    ccache "$prefix/bin/amdclang++" -x hip -O2 -std=c++17 --offload-arch=gfx1013 \
        "${flags[@]}" -I"$stage/include" -I"$stage/include/ck" \
        /workspace/helpers/ck_factory_b071.cpp -x none \
        "$stage/lib/libdevice_conv_operations.a" \
        -o "/workspace/output/factory-$mode" \
        > "/workspace/output/build-$mode.log" 2>&1
done
ccache --show-stats
