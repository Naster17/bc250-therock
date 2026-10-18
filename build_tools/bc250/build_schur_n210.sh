#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
# Same cache/settings as the qualified targeted library builds; one translation unit.
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
ccache /workspace/build/dist/rocm/bin/hipcc -O2 -fPIC -shared --offload-arch=gfx1013 \
  /workspace/helpers/schur_q_n210.hip -o /workspace/helpers/schur_q_n210.so
ccache --show-stats
