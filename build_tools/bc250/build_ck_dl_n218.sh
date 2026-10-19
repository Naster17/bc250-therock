#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
prefix=/workspace/build/dist/rocm
stage=/workspace/build/math-libs/composable_kernel/stage
# Compile only this direct scalar DL consumer, using the installed CK headers.
ccache "$prefix/bin/amdclang++" -x hip -O2 -std=c++17 --offload-arch=gfx1013 \
    -I"$stage/include" -I"$stage/include/ck" \
    /workspace/helpers/ck_dl_n218.cpp -o /workspace/output/ck_dl \
    > /workspace/output/ck-dl-build.log 2>&1
ccache --show-stats
