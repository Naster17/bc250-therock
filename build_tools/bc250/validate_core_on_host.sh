#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run after core-build.exit reports success. No old ROCm prefixes are mounted.
set -eu

workspace="${1:?usage: validate_core_on_host.sh <workspace-directory>}"
image="${BC250_BUILD_IMAGE:-localhost/bc250-rocm-build:10.2-ccache}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/core-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
mkdir -p "$workspace/build/bc250-validation"
dmesg > "$workspace/logs/kernel-before-core-validation.log"

podman run --rm --network none \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/build:/workspace/build:rw" \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        export ROCM_PATH="$prefix"
        export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        hipcc -O2 --offload-arch=gfx1013 \
            /workspace/source/build_tools/bc250/hip_smoke.cpp \
            -o /workspace/build/bc250-validation/hip_smoke
        llvm-objdump --offloading /workspace/build/bc250-validation/hip_smoke
    ' > "$workspace/logs/hip-smoke-compile.log" 2>&1

podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/build:/workspace/build:ro" \
    -e HSA_ENABLE_SDMA=0 \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        timeout --kill-after=10s 180s /workspace/build/bc250-validation/hip_smoke
    ' > "$workspace/logs/hip-smoke-sdma0.log" 2>&1

dmesg > "$workspace/logs/kernel-after-core-validation.log"
cat "$workspace/logs/hip-smoke-sdma0.log"
echo CORE_GFX1013_VALIDATION_PASS
