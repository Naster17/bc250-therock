#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_hipfile_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
mkdir -p "$workspace/logs/hipfile-data"
dmesg > "$workspace/logs/kernel-before-hipfile.log"
for example in basics/roundtrip-verify async/roundtrip-async \
    async/roundtrip-async-nonblocking-stream async/roundtrip-async-multi-stream-registered \
    batch/batch-roundtrip; do
    name="${example##*/}"
    set --
    if test "$name" = batch-roundtrip; then
        set -- 1048577 128
    fi
    podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
        -v "$workspace/build:/workspace/build:ro" \
        -v "$workspace/logs/hipfile-data:/workspace/hipfile-data:rw" \
        -e HSA_ENABLE_SDMA=0 -e HIPFILE_FORCE_COMPAT_MODE=true \
        -e LD_LIBRARY_PATH=/workspace/build/storage-libs/hipfile/dist/lib:/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
        localhost/bc250-rocm-build:10.2-ccache \
        timeout --kill-after=10s 120s \
        "/workspace/build/storage-libs/hipfile/build/examples/$example" \
        "/workspace/hipfile-data/$name-input" "/workspace/hipfile-data/$name-output" \
        "$@" 0 > "$workspace/logs/hipfile-$name.log" 2>&1
    cmp "$workspace/logs/hipfile-data/$name-input" "$workspace/logs/hipfile-data/$name-output"
    echo "HIPFILE_CPU_VERIFIED_ROUNDTRIP_PASS $name"
done
dmesg > "$workspace/logs/kernel-after-hipfile.log"
echo HIPFILE_NATIVE_GFX1013_FALLBACK_PASS
