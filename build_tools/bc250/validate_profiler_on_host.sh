#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_profiler_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/expanded-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
echo 1 > "$workspace/logs/profiler-validation.exit"
image=localhost/bc250-rocm-build:10.2-ccache
mkdir -p "$workspace/logs/profiler-trace"
dmesg > "$workspace/logs/kernel-before-profiler.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/build:/workspace/build:ro" \
    -v "$workspace/logs/profiler-trace:/workspace/trace:rw" \
    -e HSA_ENABLE_SDMA=0 \
    -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    "$image" timeout --kill-after=10s 180s \
    /workspace/build/dist/rocm/bin/rocprofv3 \
    --hip-trace --kernel-trace --memory-copy-trace --output-format json \
    --output-directory /workspace/trace --output-file bc250 \
    -- /workspace/build/bc250-validation/hip_smoke \
    > "$workspace/logs/profiler-validation.log" 2>&1
status=$?
set -e
if test "$status" -ne 0; then
    echo "$status" > "$workspace/logs/profiler-validation.exit"
fi
dmesg > "$workspace/logs/kernel-after-profiler.log"
test "$status" -eq 0
grep -F HIP_SMOKE_PASS "$workspace/logs/profiler-validation.log"
python3 "$workspace/helpers/validate_profiler_trace.py" \
    "$workspace/logs/profiler-trace/bc250_results.json" \
    >> "$workspace/logs/profiler-validation.log" 2>&1
echo 0 > "$workspace/logs/profiler-validation.exit"
