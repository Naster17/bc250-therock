#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
prefix=/workspace/build/dist/rocm
miopen_source=/workspace/source/rocm-libraries/projects/miopen
output=/workspace/build/bc250-a02-profile
mkdir -p "$output"
args=(-x hip -O2 -std=c++17 --offload-arch=gfx1013 --cuda-device-only -S
    -DCK_AMD_GPU_GFX1013
    -I"$miopen_source/src/legacy_composable_kernel/composable_kernel/include"
    -I"$miopen_source/src/kernels"
    /workspace/helpers/miopen_a02_native_profile.cpp)
ccache "$prefix/bin/amdclang++" "${args[@]}" -o "$output/native.s"
# Compiler metadata, not an assumed architecture label, must establish wave32.
grep -E 'wavefront_size: *32' "$output/native.s"
if ccache "$prefix/bin/amdclang++" "${args[@]}" -DCK_USE_AMD_XDLOPS=1 \
    -o "$output/rejected.s" > "$output/rejected.log" 2>&1; then
    echo 'Unexpected XDLOPS acceptance' >&2
    exit 1
fi
grep -F 'Native gfx1013 does not support XDLOPS' "$output/rejected.log"
echo A02_NATIVE_PROFILE_PASS
