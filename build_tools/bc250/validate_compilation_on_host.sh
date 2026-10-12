#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_compilation_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/torch-validation.exit")" = 0
test "$(cat "$workspace/logs/triton-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active

# Native/compiler validation goes through make before running tests.
podman run --rm --network none \
    -v "$workspace/triton-work:/workspace/triton-work:rw" \
    -v "$workspace/triton-deps:/workspace/triton-deps:ro" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$workspace/logs:/workspace/logs:rw" \
    localhost/bc250-triton-runtime:gfx1013 \
    bash /workspace/helpers/validate_triton_lowering.sh \
    /workspace/triton-work /workspace/triton-deps /workspace/logs/triton-lowering \
    > "$workspace/logs/triton-lowering.log" 2>&1

mkdir -p "$workspace/cache/triton-runtime" "$workspace/cache/torch-inductor"
dmesg > "$workspace/logs/kernel-before-compiler-validation.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$workspace/pytorch-work:/workspace/pytorch-work:ro" \
    -v "$workspace/triton-work:/workspace/triton-work:ro" \
    -v "$workspace/build:/workspace/build:ro" \
    -v "$workspace/cache/triton-runtime:/workspace/triton-cache:rw" \
    -v "$workspace/cache/torch-inductor:/workspace/inductor-cache:rw" \
    -v "$workspace/logs:/workspace/logs:rw" \
    -e HSA_ENABLE_SDMA=0 -e OMP_NUM_THREADS=2 \
    -e ROCM_PATH=/workspace/build/dist/rocm \
    -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    -e PYTHONPATH=/workspace/triton-work/python \
    -e TRITON_BACKENDS_IN_TREE=1 \
    -e TRITON_CACHE_DIR=/workspace/triton-cache \
    -e TORCHINDUCTOR_CACHE_DIR=/workspace/inductor-cache \
    -e TORCHINDUCTOR_COMPILE_THREADS=4 \
    localhost/bc250-pytorch-runtime:2.13-gfx1013 \
    timeout --kill-after=10s 900s python /workspace/helpers/validate_torch_compile.py \
    --triton-source /workspace/triton-work --output /workspace/logs/torch-compile-validation.json \
    > "$workspace/logs/torch-compile-validation.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/torch-compile-validation.exit"
dmesg > "$workspace/logs/kernel-after-compiler-validation.log"
test "$status" -eq 0
