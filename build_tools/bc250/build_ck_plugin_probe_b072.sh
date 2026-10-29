#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
prefix=/workspace/build/dist/rocm
mi=/workspace/source/rocm-libraries/projects/miopen
cfg=/workspace/build/ml-libs/MIOpen/build
ccache "$prefix/bin/amdclang++" -x hip -O2 -DNDEBUG -DMIOPEN_BETA_API=1 -std=c++20 --offload-arch=gfx1013 \
    -I"$cfg/include" -I"$mi/include" -I"$mi/src/include" -I"$prefix/include" \
    -I/workspace/build/third-party/nlohmann-json/dist/include \
    -I/workspace/build/third-party/sysdeps/linux/sqlite3/build/dist/lib/rocm_sysdeps/include \
    /workspace/helpers/miopen_ck_b072_probe.cpp -L"$prefix/lib" -lMIOpen -ldl \
    -o /workspace/output/probe > /workspace/output/build-probe.log 2>&1
ccache --show-stats
