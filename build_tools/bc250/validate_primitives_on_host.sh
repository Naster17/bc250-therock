#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_primitives_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/math-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
dmesg > "$workspace/logs/kernel-before-primitives-validation.log"

for executable in test_device_reduce test_device_scan test_hipcub_basic test_hipcub_block_reduce; do
    case "$executable" in
        test_device_reduce)
            filter='RocprimDeviceReduceTests/Float_Float_WarpReduce_2048.*:RocprimDeviceReduceTests/Int_Int_Raking_512.*'
            ;;
        test_device_scan)
            filter='RocprimDeviceScanTests/3.*:RocprimDeviceScanTests/5.*'
            ;;
        *) filter='*' ;;
    esac
    log="$workspace/logs/primitives-$executable.log"
    podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
        -v "$workspace/build:/workspace/build:ro" -e HSA_ENABLE_SDMA=0 \
        -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
        localhost/bc250-rocm-build:10.2-ccache \
        timeout --kill-after=10s 300s "/workspace/build/dist/rocm/bin/$executable" \
        "--gtest_filter=$filter" > "$log" 2>&1
    # A stale filter must not silently turn a hardware gate into zero tests.
    grep -E '\[  PASSED  \] [1-9][0-9]* tests?\.' "$log"
    echo "PRIMITIVES_PASS $executable"
done
dmesg > "$workspace/logs/kernel-after-primitives-validation.log"
echo PRIMITIVES_GFX1013_VALIDATION_PASS
