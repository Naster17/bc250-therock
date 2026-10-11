#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Continue authorized stages after the independent core build completes.
set -eu

workspace="${1:?usage: continue_on_host.sh <workspace-directory>}"
image="${BC250_BUILD_IMAGE:-localhost/bc250-rocm-build:10.2-ccache}"
test "$(id -u)" -eq 0

phase=waiting-for-core
trap 'status=$?; echo "$phase $status" > "$workspace/logs/continuation.exit"' EXIT
echo "$phase" > "$workspace/logs/continuation.phase"
while ! test -f "$workspace/logs/core-build.exit"; do
    sleep 15
done
test "$(cat "$workspace/logs/core-build.exit")" = 0

phase=validating-core
echo "$phase" > "$workspace/logs/continuation.phase"
sh "$workspace/helpers/validate_core_on_host.sh" "$workspace" \
    > "$workspace/logs/core-validation.log" 2>&1

phase=building-math
echo "$phase" > "$workspace/logs/continuation.phase"
# AMD SMI generates version headers and updates its Rust bindings in-tree.
# Give it a disposable copy without allowing writes to the source snapshot.
mkdir -p "$workspace/build/source-copies/amdsmi"
cp -a "$workspace/source/rocm-systems/projects/amdsmi/." \
    "$workspace/build/source-copies/amdsmi/"
set +e
podman run --rm --name bc250-rocm-math-build --network host \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/build/source-copies/amdsmi:/workspace/source/rocm-systems/projects/amdsmi:rw" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/cache:/workspace/cache:rw" \
    -e CMAKE_BUILD_PARALLEL_LEVEL="${BC250_BUILD_JOBS:-16}" \
    -e BC250_FLANG_JOBS="${BC250_FLANG_JOBS:-4}" \
    -e BC250_CK_JOBS="${BC250_CK_JOBS:-6}" \
    -e BC250_FFT_JOBS="${BC250_FFT_JOBS:-16}" \
    -e BC250_CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}" \
    "$image" bash /workspace/helpers/build_math.sh /workspace/source /workspace/build \
    > "$workspace/logs/math-build.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/math-build.exit"
test "$status" -eq 0
phase=math-built-awaiting-validation
echo "$phase" > "$workspace/logs/continuation.phase"
