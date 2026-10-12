#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_rocalution_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
filter='cg/parameterized_cg.cg_float/0:cg/parameterized_cg.cg_double/0:cg/parameterized_cg.cg_float/16:cg/parameterized_cg.cg_double/16'
dmesg > "$workspace/logs/kernel-before-rocalution.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/build:/workspace/build:ro" -e HSA_ENABLE_SDMA=0 -e OMP_NUM_THREADS=4 \
    -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    localhost/bc250-rocm-build:10.2-ccache \
    timeout --kill-after=10s 120s \
    /workspace/build/math-libs/rocALUTION/stage/bin/rocalution-test \
    "--gtest_filter=$filter" > "$workspace/logs/rocalution-validation.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/rocalution-validation.exit"
dmesg > "$workspace/logs/kernel-after-rocalution.log"
test "$status" -eq 0
grep -F '[  PASSED  ] 4 tests.' "$workspace/logs/rocalution-validation.log"
if grep -F '[  SKIPPED ]' "$workspace/logs/rocalution-validation.log"; then
    exit 1
fi
