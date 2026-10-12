#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_openmp_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
mkdir -p "$workspace/build/bc250-validation"
image=localhost/bc250-rocm-build:10.2-ccache
podman run --rm --network none \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        "$prefix/bin/amdclang" -O2 -fopenmp --offload-arch=gfx1013 \
            /workspace/helpers/openmp_smoke.c \
            -o /workspace/build/bc250-validation/openmp_smoke
    ' > "$workspace/logs/openmp-compile.log" 2>&1
dmesg > "$workspace/logs/kernel-before-openmp.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/build:/workspace/build:ro" \
    -e HSA_ENABLE_SDMA=0 -e OMP_TARGET_OFFLOAD=MANDATORY -e OMP_NUM_THREADS=4 \
    -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    "$image" timeout --kill-after=10s 120s /workspace/build/bc250-validation/openmp_smoke \
    > "$workspace/logs/openmp-validation.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/openmp-validation.exit"
dmesg > "$workspace/logs/kernel-after-openmp.log"
test "$status" -eq 0
