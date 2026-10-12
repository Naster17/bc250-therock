#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: run_expanded_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
image="${BC250_BUILD_IMAGE:-localhost/bc250-rocm-build:10.2-ccache}"
test "$(cat "$workspace/logs/math-build.exit")" = 0
rm -f "$workspace/logs/expanded-build.exit"
set +e
podman run --rm --name bc250-rocm-expanded-build --network host \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/build/source-copies/amdsmi:/workspace/source/rocm-systems/projects/amdsmi:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/cache:/workspace/cache:rw" \
    -e BC250_BUILD_JOBS="${BC250_BUILD_JOBS:-16}" \
    -e BC250_HIPTENSOR_JOBS="${BC250_HIPTENSOR_JOBS:-12}" \
    -e BC250_CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}" \
    "$image" bash /workspace/helpers/build_expanded.sh /workspace/source /workspace/build \
    > "$workspace/logs/expanded-build.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/expanded-build.exit"
exit "$status"
