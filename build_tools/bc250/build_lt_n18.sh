#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G CMAKE_BUILD_PARALLEL_LEVEL=16 OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
cmake --build /workspace/build --target hipBLASLt+dist --parallel 16
ccache --show-stats
