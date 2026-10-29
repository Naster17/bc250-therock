#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
prefix=/workspace/build/dist/rocm
stage=/workspace/build/math-libs/composable_kernel/stage
# The qualified archive lacks this group-major FP16 factory. Build only its
# unchanged matching source into the isolated candidate; never modify that archive.
ccache "$prefix/bin/amdclang++" -x hip -O2 -std=c++17 --offload-arch=gfx1013 -DDL_KERNELS \
    -I"$stage/include" -I"$stage/include/ck" -c /workspace/inputs/gnhwc-f16-instance.cpp \
    -o /workspace/output/gnhwc-f16-instance.o > /workspace/output/build-gnhwc.log 2>&1
for mode in nodl dl; do
    objects=()
    if [[ "$mode" == dl ]]; then objects+=(/workspace/output/gnhwc-f16-instance.o); fi
    flags=()
    if [[ "$mode" == dl ]]; then flags+=(-DDL_KERNELS); fi
    ccache "$prefix/bin/amdclang++" -x hip -O2 -std=c++17 --offload-arch=gfx1013 \
        "${flags[@]}" -I"$stage/include" -I"$stage/include/ck" -I/workspace/build/third-party/nlohmann-json/dist/include \
        /workspace/inputs/ck_factory_b079.cpp -x none "${objects[@]}" \
        "$stage/lib/libdevice_conv_operations.a" \
        -o "/workspace/output/factory-$mode" \
        > "/workspace/output/build-$mode.log" 2>&1
done
ccache --show-stats
