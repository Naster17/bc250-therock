#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
prefix=/workspace/build/dist/rocm
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/rccl-b034-config --local-path /workspace/cache/data --no-reset-stats)"
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
ccache "$prefix/lib/llvm/bin/clang++" -x hip --offload-arch=gfx1013 \
  --rocm-path="$prefix" --hip-path="$prefix" -O2 -std=c++17 \
  /workspace/helpers/rccl_readiness_b034.cpp \
  -I/workspace/rccl/install-r1/include -L/workspace/rccl/install-r1/lib \
  -Wl,-rpath-link,"$prefix/lib" -lrccl -o /workspace/rccl/readiness-r1
ccache --show-stats
