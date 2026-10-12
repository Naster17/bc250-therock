#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_rocalution_diagnostic_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/expanded-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
echo 1 > "$workspace/logs/rocalution-diagnostic.exit"
image=localhost/bc250-rocm-build:10.2-ccache
podman run --rm --network none \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        stage=/workspace/build/math-libs/rocALUTION/stage
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        "$prefix/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1013 \
            -I"$stage/include" -L"$stage/lib" /workspace/helpers/rocalution_diagnostic.cpp \
            -lrocalution -o /workspace/build/bc250-validation/rocalution_diagnostic
    ' > "$workspace/logs/rocalution-diagnostic-compile.log" 2>&1
dmesg > "$workspace/logs/kernel-before-rocalution-diagnostic.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/build:/workspace/build:ro" -e HSA_ENABLE_SDMA=0 -e OMP_NUM_THREADS=4 \
    -e LD_LIBRARY_PATH=/workspace/build/math-libs/rocALUTION/dist/lib:/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    "$image" \
    timeout --kill-after=10s 120s \
    /workspace/build/bc250-validation/rocalution_diagnostic > "$workspace/logs/rocalution-diagnostic.log" 2>&1
status=$?
set -e
if test "$status" -ne 0; then
    echo "$status" > "$workspace/logs/rocalution-diagnostic.exit"
fi
dmesg > "$workspace/logs/kernel-after-rocalution-diagnostic.log"
test "$status" -eq 0
test "$(grep -c 'CG_DIAGNOSTIC.*pass=1' "$workspace/logs/rocalution-diagnostic.log")" -eq 40
echo 0 > "$workspace/logs/rocalution-diagnostic.exit"
