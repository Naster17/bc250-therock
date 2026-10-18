#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
compile() {
    local number="$1" library="$2"
    ccache /workspace/build/dist/rocm/bin/amdclang++ -x hip -O2 -std=c++17 \
        --offload-arch=gfx1013 -DN217_LIBRARY="$number" \
        -I/workspace/build/dist/rocm/include \
        /workspace/helpers/sorting_n217.cpp -o "/workspace/output/$library"
}
# Three independent units, below the documented 16-job native-library limit.
compile 0 rocprim > /workspace/output/rocprim-build.log 2>&1 &
p0=$!
compile 1 hipcub > /workspace/output/hipcub-build.log 2>&1 &
p1=$!
compile 2 thrust > /workspace/output/thrust-build.log 2>&1 &
p2=$!
status=0
for pid in "$p0" "$p1" "$p2"; do wait "$pid" || status=1; done
ccache --show-stats
exit "$status"
