#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_rpp_audio_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/expanded-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
echo 1 > "$workspace/logs/rpp-audio-validation.exit"
image=localhost/bc250-rocm-build:10.2-ccache
podman run --rm --network none \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        stage=/workspace/build/cv-libs/rpp/stage
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        "$prefix/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1013 -DAUDIO_SUPPORT \
            -I"$stage/include" -L"$stage/lib" /workspace/helpers/rpp_audio_smoke.cpp \
            -lrpp -o /workspace/build/bc250-validation/rpp_audio_smoke
    ' > "$workspace/logs/rpp-audio-compile.log" 2>&1
dmesg > "$workspace/logs/kernel-before-rpp-audio.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/build:/workspace/build:ro" -e HSA_ENABLE_SDMA=0 -e OMP_NUM_THREADS=4 \
    -e LD_LIBRARY_PATH=/workspace/build/cv-libs/rpp/stage/lib:/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    "$image" timeout --kill-after=10s 120s /workspace/build/bc250-validation/rpp_audio_smoke \
    > "$workspace/logs/rpp-audio-validation.log" 2>&1
status=$?
set -e
if test "$status" -ne 0; then
    echo "$status" > "$workspace/logs/rpp-audio-validation.exit"
fi
dmesg > "$workspace/logs/kernel-after-rpp-audio.log"
test "$status" -eq 0
grep -F 'RPP_AUDIO_NATIVE_GFX1013_PASS' "$workspace/logs/rpp-audio-validation.log"
echo 0 > "$workspace/logs/rpp-audio-validation.exit"
