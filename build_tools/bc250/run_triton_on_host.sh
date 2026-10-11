#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: run_triton_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
image="${BC250_TRITON_BUILD_IMAGE:-localhost/bc250-triton-build:gfx1013}"
if podman container exists bc250-triton-build; then
    test "$(podman inspect --format '{{.State.Running}}' bc250-triton-build)" = false
    podman rm bc250-triton-build
fi
podman image inspect "$image" > "$workspace/logs/triton-build-image.json"
rm -f "$workspace/logs/triton-build.exit"
set +e
podman run --name bc250-triton-build --network none \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$workspace/triton-source:/workspace/triton-source:ro" \
    -v "$workspace/triton-work:/workspace/triton-work:rw" \
    -v "$workspace/triton-deps:/workspace/triton-deps:ro" \
    -v "$workspace/cache:/workspace/cache:rw" \
    -e BC250_TRITON_JOBS="${BC250_TRITON_JOBS:-4}" \
    -e BC250_CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}" \
    "$image" nice -n 10 bash /workspace/helpers/build_triton.sh \
    /workspace/triton-source /workspace/triton-work /workspace/triton-deps \
    /workspace/source > "$workspace/logs/triton-build.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/triton-build.exit"
exit "$status"
