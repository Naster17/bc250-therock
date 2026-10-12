#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Verify the new ROCm libraries with the existing, independently copied Torch.
# This does not activate a release or alter the working installation.
set -eu
workspace="${1:?usage: finish_candidate_torch_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
result="$workspace/logs/candidate-torch-validation.exit"
echo running > "$result"
trap 'status=$?; echo "$status" > "$result"' EXIT
while podman container exists bc250-rocm-expanded-build || \
    pgrep -f "^sh $workspace/helpers/finish_expanded_validation_on_host.sh" >/dev/null; do
    sleep 15
done
test "$(cat "$workspace/logs/expanded-build.exit")" = 0
test "$(cat "$workspace/logs/expanded-validation.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
release="$(readlink -f /opt/bc250-stack/current)"
image="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$release/runtime.json")"
uid="$(stat -c %u "$workspace")"
gid="$(stat -c %g "$workspace")"
cache="$workspace/cache/candidate-runtime"
logs="$workspace/logs/candidate-runtime"
mkdir -p "$cache" "$logs"
chown "$uid:$gid" "$cache" "$logs"

# Honor the Triton source instructions before its native/compiler GPU checks.
podman run --rm --network none \
    -v "$workspace/triton-work:/workspace/triton-work:rw" \
    -v "$workspace/triton-deps:/workspace/triton-deps:ro" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$logs:/workspace/logs:rw" \
    localhost/bc250-triton-runtime:gfx1013 \
    bash /workspace/helpers/validate_triton_lowering.sh \
    /workspace/triton-work /workspace/triton-deps /workspace/logs/triton-lowering \
    > "$logs/triton-lowering.log" 2>&1

dmesg > "$logs/kernel-before.log"
podman run --rm --pull=never --network none \
    --device /dev/kfd --device /dev/dri/renderD128 \
    --user "$uid:$gid" \
    --group-add "$(stat -c %g /dev/kfd)" \
    --group-add "$(stat -c %g /dev/dri/renderD128)" \
    -v "$workspace/build/dist/rocm:/workspace/build/dist/rocm:ro" \
    -v "$release/torch:/workspace/pytorch-work:ro" \
    -v "$workspace/triton-work:/workspace/triton-work:ro" \
    -v "$workspace/helpers:/workspace/helpers:ro" \
    -v "$cache:/workspace/user-cache:rw" -v "$logs:/workspace/logs:rw" \
    -e HSA_ENABLE_SDMA=0 -e OMP_NUM_THREADS=4 \
    -e ROCM_PATH=/workspace/build/dist/rocm -e HIP_PATH=/workspace/build/dist/rocm \
    -e LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib \
    -e PATH=/workspace/build/dist/rocm/bin:/workspace/build/dist/rocm/lib/llvm/bin:/opt/torch-venv/bin:/usr/local/therock-tools/bin:/usr/bin:/bin \
    -e HOME=/workspace/user-cache -e PYTHONDONTWRITEBYTECODE=1 \
    -e TRITON_CACHE_DIR=/workspace/user-cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/workspace/user-cache/inductor \
    -e TORCHINDUCTOR_COMPILE_THREADS=4 \
    "$image" bash -c '
        set -euo pipefail
        timeout --kill-after=10s 900s python /workspace/helpers/validate_torch.py \
            --output /workspace/logs/eager.json > /workspace/logs/eager.log 2>&1
        echo 0 > /workspace/logs/eager.exit
        timeout --kill-after=10s 900s python /workspace/helpers/validate_torch_compile.py \
            --triton-source /workspace/triton-work --output /workspace/logs/compiler.json \
            > /workspace/logs/compiler.log 2>&1
        echo 0 > /workspace/logs/compiler.exit
        timeout --kill-after=10s 900s python /workspace/helpers/validate_stability.py \
            --duration 600 --output /workspace/logs/stability.json \
            > /workspace/logs/stability.log 2>&1
        echo 0 > /workspace/logs/stability.exit
    ' > "$logs/container.log" 2>&1
dmesg > "$logs/kernel-after.log"
echo CANDIDATE_TORCH_AND_STABILITY_PASS
