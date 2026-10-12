#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Use an immutable release to isolate sparse failures from development builds.
set -eu
workspace="${1:?usage: diagnose_sparse_on_host.sh <workspace-directory> [rocm-prefix]}"
prefix="${2:-/opt/bc250-stack/current/rocm}"
test "$(id -u)" -eq 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
image=localhost/bc250-rocm-build:10.2-ccache
mkdir -p "$workspace/probes"
echo 1 > "$workspace/logs/sparse-diagnostic.exit"
podman run --rm --network none \
    -v "$prefix:/workspace/rocm:ro" \
    -v "$workspace/probes:/workspace/probes:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    "$image" bash -c '
        set -euo pipefail
        export LD_LIBRARY_PATH=/workspace/rocm/lib:/workspace/rocm/lib/llvm/lib
        /workspace/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx1013 \
            /workspace/helpers/rocsparse_legacy_spmv.cpp \
            -lrocsparse -o /workspace/probes/rocsparse_legacy_spmv
    ' > "$workspace/logs/sparse-diagnostic-compile.log" 2>&1
dmesg > "$workspace/logs/kernel-before-sparse-diagnostic.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$prefix:/workspace/rocm:ro" \
    -v "$workspace/probes:/workspace/probes:ro" \
    -e HSA_ENABLE_SDMA=0 \
    -e LD_LIBRARY_PATH=/workspace/rocm/lib:/workspace/rocm/lib/llvm/lib \
    "$image" timeout --kill-after=10s 120s /workspace/probes/rocsparse_legacy_spmv \
    > "$workspace/logs/sparse-diagnostic.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/sparse-diagnostic.exit"
dmesg > "$workspace/logs/kernel-after-sparse-diagnostic.log"
cat "$workspace/logs/sparse-diagnostic.log"
exit "$status"
