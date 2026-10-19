#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
build=/workspace/build/math-libs/BLAS/rocBLAS/build
ninja -C "$build" -n rocblas > /workspace/candidate/build-plan.log
cmake --build "$build" --target rocblas --parallel 16
cp "$build/library/src/librocblas.so.5.8" /workspace/candidate/librocblas.so.5.8
/workspace/build/dist/rocm/lib/llvm/bin/llvm-strip --strip-unneeded /workspace/candidate/librocblas.so.5.8
ln -s librocblas.so.5.8 /workspace/candidate/librocblas.so.5
ln -s librocblas.so.5 /workspace/candidate/librocblas.so
ccache --show-stats
