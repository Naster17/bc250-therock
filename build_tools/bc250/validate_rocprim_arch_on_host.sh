#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Validate a committed header overlay before rebuilding all header consumers.
set -eu
workspace="${1:?usage: validate_rocprim_arch_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/expanded-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
echo 1 > "$workspace/logs/rocprim-arch-validation.exit"
image=localhost/bc250-rocm-build:10.2-ccache
podman run --rm --network none \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/arch-port:/workspace/arch-port:ro" \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        for variant in main hipccl2 hipccl3; do
            if [[ "$variant" == main ]]; then
                root=projects/rocprim
            else
                root="projects/hipccl/$variant/rocprim"
            fi
            "$prefix/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1013 \
                -I"/workspace/arch-port/$root/rocprim/include" \
                -I"/workspace/source/rocm-libraries/$root/rocprim/include" \
                /workspace/helpers/rocprim_arch_smoke.cpp \
                -o "/workspace/build/bc250-validation/rocprim_arch_$variant"
        done
    ' > "$workspace/logs/rocprim-arch-compile.log" 2>&1
dmesg > "$workspace/logs/kernel-before-rocprim-arch.log"
for variant in main hipccl2 hipccl3; do
    podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
        -v "$workspace/build:/workspace/build:ro" -e HSA_ENABLE_SDMA=0 \
        -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
        "$image" timeout --kill-after=10s 120s \
        "/workspace/build/bc250-validation/rocprim_arch_$variant" \
        > "$workspace/logs/rocprim-arch-$variant.log" 2>&1
    grep -F ROCPRIM_NATIVE_ARCH_DISPATCH_PASS "$workspace/logs/rocprim-arch-$variant.log"
done
dmesg > "$workspace/logs/kernel-after-rocprim-arch.log"
echo 0 > "$workspace/logs/rocprim-arch-validation.exit"
