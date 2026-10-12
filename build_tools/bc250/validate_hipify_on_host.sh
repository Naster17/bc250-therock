#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Translate and compile without GPU access, then execute after other GPU gates.
set -eu
workspace="${1:?usage: validate_hipify_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
result="$workspace/logs/hipify-validation.exit"
echo running > "$result"
trap 'status=$?; echo "$status" > "$result"' EXIT
image=localhost/bc250-rocm-build:10.2-ccache
podman run --rm --network none \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        output=/workspace/build/bc250-validation
        mkdir -p "$output"
        "$prefix/bin/hipify-clang" --version
        "$prefix/bin/hipify-perl" /workspace/helpers/hipify_smoke.cu > "$output/hipify_smoke.cpp"
        grep -F "hip/hip_runtime.h" "$output/hipify_smoke.cpp"
        grep -F hipMemcpyAsync "$output/hipify_smoke.cpp"
        "$prefix/bin/hipcc" -O2 --offload-arch=gfx1013 "$output/hipify_smoke.cpp" \
            -o "$output/hipify_smoke"
    ' > "$workspace/logs/hipify-compile.log" 2>&1
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/build:/workspace/build:ro" -e HSA_ENABLE_SDMA=0 \
    -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    "$image" timeout --kill-after=10s 120s /workspace/build/bc250-validation/hipify_smoke \
    > "$workspace/logs/hipify-validation.log" 2>&1
grep -F HIPIFY_TRANSLATED_KERNEL_PASS "$workspace/logs/hipify-validation.log"
