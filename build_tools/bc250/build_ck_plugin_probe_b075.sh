#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
prefix=/workspace/build/dist/rocm
mi=/workspace/source/rocm-libraries/projects/miopen
cfg=/workspace/build/ml-libs/MIOpen/build
for kind in b075; do
ccache "$prefix/bin/amdclang++" -x hip -O2 -DNDEBUG -DMIOPEN_BETA_API=1 -std=c++20 --offload-arch=gfx1013 \
    -I"$cfg/include" -I"$mi/include" -I"$mi/src/include" -I"$prefix/include" \
    -I/workspace/build/third-party/nlohmann-json/dist/include \
    -I/workspace/build/third-party/sysdeps/linux/sqlite3/build/dist/lib/rocm_sysdeps/include \
    /workspace/inputs/miopen_ck_${kind}_probe.cpp -L"$prefix/lib" -lMIOpen -ldl \
    -o /workspace/output/probe-${kind} > /workspace/output/build-probe-${kind}.log 2>&1
done
ccache --show-stats
