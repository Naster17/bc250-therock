#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail
source_dir="${1:?source directory}"
build_dir="${2:?build directory}"
prefix="$build_dir/dist/rocm"
output="$build_dir/bc250-coverage/priority2-primitives"
mkdir -p "$output"
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
export CCACHE_MAXSIZE=20G
compile() {
    local library="$1"
    ccache "$prefix/bin/amdclang++" -x hip -O2 -std=c++17 --offload-arch=gfx1013 \
        -I"$prefix/include" "$source_dir/build_tools/bc250/priority2_primitives_${library}.cpp" \
        -o "$output/$library"
}
compile rocprim &
p1=$!
compile hipcub &
p2=$!
compile thrust &
p3=$!
status=0
for pid in "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
exit "$status"
