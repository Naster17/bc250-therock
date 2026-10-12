#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_opencl_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
image=localhost/bc250-rocm-build:10.2-ccache
mkdir -p "$workspace/build/bc250-validation/opencl-icd"
printf '%s\n' /workspace/build/core/ocl-clr/dist/lib/opencl/libamdocl64.so \
    > "$workspace/build/bc250-validation/opencl-icd/bc250.icd"
podman run --rm --network none \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    "$image" bash -c '
        set -euo pipefail
        gcc -O2 /workspace/helpers/opencl_smoke.c \
            -I/workspace/source/rocm-systems/projects/clr/opencl/khronos/headers/opencl2.2 \
            -L/workspace/build/core/ocl-icd/dist/lib -lOpenCL \
            -o /workspace/build/bc250-validation/opencl_smoke
    ' > "$workspace/logs/opencl-compile.log" 2>&1
dmesg > "$workspace/logs/kernel-before-opencl.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/build:/workspace/build:ro" \
    -e HSA_ENABLE_SDMA=0 \
    -e OCL_ICD_VENDORS=/workspace/build/bc250-validation/opencl-icd \
    -e LD_LIBRARY_PATH=/workspace/build/core/ocl-icd/dist/lib:/workspace/build/core/ocl-clr/dist/lib:/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    "$image" timeout --kill-after=10s 120s /workspace/build/bc250-validation/opencl_smoke \
    > "$workspace/logs/opencl-validation.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/opencl-validation.exit"
dmesg > "$workspace/logs/kernel-after-opencl.log"
test "$status" -eq 0
