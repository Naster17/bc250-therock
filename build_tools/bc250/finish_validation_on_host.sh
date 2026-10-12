#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: finish_validation_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
rm -f "$workspace/logs/compiler-continuation.exit" \
    "$workspace/logs/torch-compile-validation.exit"
phase=waiting-for-torch
trap 'status=$?; echo "$phase $status" > "$workspace/logs/compiler-continuation.exit"' EXIT
echo "$phase" > "$workspace/logs/compiler-continuation.phase"
while ! test -f "$workspace/logs/torch-continuation.exit"; do
    sleep 15
done
test "$(cat "$workspace/logs/torch-continuation.exit")" = 'torch-validated-awaiting-deployment 0'
for component in ml-validation primitives-validation triton-build; do
    while ! test -f "$workspace/logs/$component.exit"; do
        sleep 15
    done
    test "$(cat "$workspace/logs/$component.exit")" = 0
done
phase=validating-compilation
echo "$phase" > "$workspace/logs/compiler-continuation.phase"
sh "$workspace/helpers/validate_compilation_on_host.sh" "$workspace" \
    > "$workspace/logs/compiler-validation.log" 2>&1
phase=compiler-validated-awaiting-deployment
echo "$phase" > "$workspace/logs/compiler-continuation.phase"
