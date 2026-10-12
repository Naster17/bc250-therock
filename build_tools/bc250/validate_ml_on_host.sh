#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_ml_on_host.sh <workspace-directory>}"
image="${BC250_BUILD_IMAGE:-localhost/bc250-rocm-build:10.2-ccache}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/math-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
dmesg > "$workspace/logs/kernel-before-ml-validation.log"
mkdir -p "$workspace/build/bc250-validation/ck"

# Use the upstream CPU-reference examples with the installed scalar profile.
podman run --rm --network none \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/build:/workspace/build:rw" \
    "$image" bash -c '
        set -euo pipefail
        prefix=/workspace/build/dist/rocm
        ck_stage=/workspace/build/math-libs/composable_kernel/stage
        export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
        for precision in fp32 fp16 int8; do
            "$prefix/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1013 \
                -I"$ck_stage/include" -I"$ck_stage/include/ck" \
                /workspace/source/rocm-libraries/projects/composablekernel/example/01_gemm/gemm_dl_$precision.cpp \
                -L"$ck_stage/lib" -lutility -lpthread \
                -o /workspace/build/bc250-validation/ck/gemm_dl_$precision
        done
    ' > "$workspace/logs/ck-smoke-compile.log" 2>&1

for precision in fp32 fp16 int8; do
    podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
        -v "$workspace/build:/workspace/build:ro" -e HSA_ENABLE_SDMA=0 \
        "$image" bash -c '
            set -euo pipefail
            prefix=/workspace/build/dist/rocm
            export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
            # CPU verification, signed integer inputs, no benchmark loop.
            timeout --kill-after=10s 120s \
                "/workspace/build/bc250-validation/ck/gemm_dl_$1" \
                1 1 0 256 128 64 -1 -1 -1
        ' bash "$precision" > "$workspace/logs/ck-$precision.log" 2>&1
    echo "CK_SCALAR_GEMM_PASS $precision"
done

for precision in conv convfp16 convbfp16; do
    for groups in 1 2; do
        log="$workspace/logs/miopen-$precision-groups$groups.log"
        podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
            -v "$workspace/build:/workspace/build:ro" -e HSA_ENABLE_SDMA=0 \
            -e OMP_NUM_THREADS=2 \
            "$image" bash -c '
                set -euo pipefail
                prefix=/workspace/build/dist/rocm
                export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
                export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib"
                export ROCM_PATH="$prefix" HIP_PATH="$prefix"
                timeout --kill-after=10s 300s "$prefix/bin/MIOpenDriver" "$1" \
                    -n 2 -c 4 -k 6 -H 8 -W 9 -y 3 -x 3 -p 1 -q 1 \
                    -g "$2" -F 0 -i 1 -V 1 -s 0
            ' bash "$precision" "$groups" > "$log" 2>&1
        for direction in 'Forward Convolution' 'Backward Convolution Data' 'Backward Convolution Weights'; do
            grep -F "$direction Verifies OK on CPU reference" "$log"
        done
        echo "MIOPEN_CONVOLUTION_PASS $precision groups=$groups"
    done
done
dmesg > "$workspace/logs/kernel-after-ml-validation.log"
echo ML_GFX1013_VALIDATION_PASS
