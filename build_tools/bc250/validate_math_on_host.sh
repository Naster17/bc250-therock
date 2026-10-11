#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_math_on_host.sh <workspace-directory>}"
image="${BC250_BUILD_IMAGE:-localhost/bc250-rocm-build:10.2-ccache}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/math-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
dmesg > "$workspace/logs/kernel-before-math-validation.log"

podman run --rm --network none \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$workspace/build:/workspace/build:rw" \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        export ROCM_PATH="$prefix"
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        "$prefix/bin/hipcc" -O2 --offload-arch=gfx1013 \
            /workspace/helpers/math_smoke.cpp -lhipblas -lrocblas -lrocsolver \
            -lhipfft -lrocrand -o /workspace/build/bc250-validation/math_smoke
        "$prefix/bin/hipcc" -O2 --offload-arch=gfx1013 \
            /workspace/helpers/hipblaslt_host_api.cpp -lhipblaslt \
            -o /workspace/build/bc250-validation/hipblaslt_host_api
    ' > "$workspace/logs/math-smoke-compile.log" 2>&1

for executable in math_smoke hipblaslt_host_api; do
    podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
        -v "$workspace/build:/workspace/build:ro" \
        -e HSA_ENABLE_SDMA=0 \
        "$image" bash -c '
            set -euo pipefail
            prefix=/workspace/build/dist/rocm
            export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
            timeout --kill-after=10s 300s "/workspace/build/bc250-validation/$1"
        ' bash "$executable" > "$workspace/logs/$executable.log" 2>&1
    cat "$workspace/logs/$executable.log"
done
dmesg > "$workspace/logs/kernel-after-math-validation.log"
echo MATH_GFX1013_VALIDATION_PASS
