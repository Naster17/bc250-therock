#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Recheck the allocator regression in independent processes against the release.
set -eu
workspace="${1:?usage: validate_rocalution_repeats_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
release="$(readlink -f /opt/bc250-stack/current)"
result="$workspace/logs/rocalution-repeats.exit"
echo running > "$result"
trap 'status=$?; echo "$status" > "$result"' EXIT
logs="$workspace/logs/rocalution-repeats"
mkdir -p "$logs"
dmesg > "$logs/kernel-before.log"
for iteration in 1 2 3 4 5; do
    podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
        -v "$workspace/build:/workspace/build:ro" \
        -v "$release/rocm:/workspace/build/dist/rocm:ro" \
        -e HSA_ENABLE_SDMA=0 -e OMP_NUM_THREADS=4 \
        -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
        localhost/bc250-rocm-build:10.2-ccache timeout --kill-after=10s 120s \
        /workspace/build/bc250-validation/rocalution_diagnostic \
        > "$logs/process-$iteration.log" 2>&1
    test "$(grep -c 'CG_DIAGNOSTIC.*pass=1' "$logs/process-$iteration.log")" -eq 40
    echo "ROCALUTION_FRESH_PROCESS_PASS $iteration cases=40"
done
dmesg > "$logs/kernel-after.log"
echo "ROCALUTION_FRESH_PROCESSES_PASS processes=5 cases=200 release=$release"
