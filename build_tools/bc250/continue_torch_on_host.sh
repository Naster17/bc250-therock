#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Keep the Torch tool environment in the successful container image; sources
# and compiled objects remain in the persistent working-directory mount.
set -eu
workspace="${1:?usage: continue_torch_on_host.sh <workspace-directory>}"
image="${BC250_TORCH_BUILD_IMAGE:-localhost/bc250-pytorch-build:2.13-gfx1013}"
runtime_image=localhost/bc250-pytorch-runtime:2.13-gfx1013
test "$(id -u)" -eq 0
mkdir -p "$workspace/pytorch-work"
phase=waiting-for-math
trap 'status=$?; echo "$phase $status" > "$workspace/logs/torch-continuation.exit"' EXIT
echo "$phase" > "$workspace/logs/torch-continuation.phase"
while ! test -f "$workspace/logs/math-build.exit"; do
    sleep 15
done
test "$(cat "$workspace/logs/math-build.exit")" = 0

phase=validating-math
echo "$phase" > "$workspace/logs/torch-continuation.phase"
sh "$workspace/helpers/validate_math_on_host.sh" "$workspace" \
    > "$workspace/logs/math-validation.log" 2>&1
podman image inspect "$image" > "$workspace/logs/torch-build-image.json"

phase=building-torch
echo "$phase" > "$workspace/logs/torch-continuation.phase"
if podman container exists bc250-pytorch-build; then
    test "$(podman inspect --format '{{.State.Running}}' bc250-pytorch-build)" = false
    podman rm bc250-pytorch-build
fi
set +e
podman run --name bc250-pytorch-build --network host \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$workspace/pytorch-source:/workspace/pytorch-source:ro" \
    -v "$workspace/pytorch-work:/workspace/pytorch-work:rw" \
    -v "$workspace/build:/workspace/build:ro" \
    -v "$workspace/cache:/workspace/cache:rw" \
    -e BC250_TORCH_JOBS="${BC250_TORCH_JOBS:-12}" \
    -e BC250_CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}" \
    "$image" bash /workspace/helpers/build_torch.sh \
    /workspace/pytorch-source /workspace/pytorch-work \
    /workspace/build/dist/rocm /workspace/source \
    > "$workspace/logs/torch-build.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/torch-build.exit"
test "$status" -eq 0
podman commit bc250-pytorch-build "$runtime_image" \
    > "$workspace/logs/torch-runtime-image-id"

phase=validating-torch
echo "$phase" > "$workspace/logs/torch-continuation.phase"
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
dmesg > "$workspace/logs/kernel-before-torch-validation.log"
set +e
podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$workspace/pytorch-work:/workspace/pytorch-work:ro" \
    -v "$workspace/build:/workspace/build:ro" \
    -v "$workspace/logs:/workspace/logs:rw" \
    -e HSA_ENABLE_SDMA=0 \
    -e ROCM_PATH=/workspace/build/dist/rocm \
    -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    -e OMP_NUM_THREADS=4 \
    "$runtime_image" timeout --kill-after=10s 900s \
    python /workspace/helpers/validate_torch.py --output /workspace/logs/torch-validation.json \
    > "$workspace/logs/torch-validation.log" 2>&1
status=$?
set -e
echo "$status" > "$workspace/logs/torch-validation.exit"
dmesg > "$workspace/logs/kernel-after-torch-validation.log"
test "$status" -eq 0
phase=torch-validated-awaiting-deployment
echo "$phase" > "$workspace/logs/torch-continuation.phase"
