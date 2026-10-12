#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_hipdnn_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/expanded-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
image=localhost/bc250-rocm-build:10.2-ccache
prefix=/workspace/build/dist/rocm
source=/workspace/source/rocm-libraries/dnn-providers
dmesg > "$workspace/logs/kernel-before-hipdnn.log"

run_tests() {
    label="$1"
    expected="$2"
    shift 2
    set +e
    podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
        -v "$workspace/source:/workspace/source:ro" \
        -v "$workspace/build:/workspace/build:ro" \
        -v "$workspace/logs:/workspace/logs:rw" \
        -e HSA_ENABLE_SDMA=0 -e OMP_NUM_THREADS=4 \
        -e ROCM_PATH="$prefix" -e HIP_PATH="$prefix" \
        -e LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib" \
        "$image" timeout --kill-after=10s 600s "$@" \
        "--gtest_output=xml:/workspace/logs/$label.xml" \
        > "$workspace/logs/$label.log" 2>&1
    status=$?
    set -e
    echo "$status" > "$workspace/logs/$label.exit"
    test "$status" -eq 0
    # An exit status of zero also covers no-device and all-skipped runs upstream.
    python3 - "$workspace/logs/$label.xml" "$expected" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
assert int(root.attrib["tests"]) == int(sys.argv[2]), root.attrib
for field in ("failures", "disabled", "errors"):
    assert int(root.attrib.get(field, 0)) == 0, root.attrib
for case in root.iter("testcase"):
    assert case.attrib.get("result") == "completed", case.attrib
    assert case.find("skipped") is None, case.attrib
print("HIPDNN_GPU_TESTS_PASS", root.attrib["tests"])
PY
}

run_tests hipkernel-pass-by-value 5 \
    "$prefix/bin/hip_kernel_provider_integration_tests" \
    '--gtest_filter=IntegrationGpuPassByValue.*'

run_tests hipkernel-cpu-reference 9 \
    "$prefix/bin/hipdnn_integration_tests" \
    --test-article "$prefix/lib/hipdnn_plugins/engines/libhip_kernel_provider.so" \
    --test-engine HIP_MLOPS_ENGINE \
    --test-config "$source/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml" \
    --reference-executor cpu --verification-mode cpu --validator cpu \
    --no-bundles --fail-on-unsupported \
    '--gtest_filter=Quick/IntegrationGpuLayernormPure4D*.Correctness/0:Smoke/IntegrationGpuRMSNorm*.Correctness/0:Smoke/IntegrationGpuResampleForward*.Correctness/0'

run_tests miopen-provider-cpu-reference 9 \
    "$prefix/bin/hipdnn_integration_tests" \
    --test-article "$prefix/lib/hipdnn_plugins/engines/libmiopen_plugin.so" \
    --test-engine MIOPEN_ENGINE \
    --reference-executor cpu --verification-mode cpu --validator cpu \
    --no-bundles --fail-on-unsupported \
    '--gtest_filter=Smoke/IntegrationGpuConvFwd2d*.Correctness/0:Smoke/IntegrationGpuConvBwdData2d*.Correctness/0:Smoke/IntegrationGpuConvWrw2d*.Correctness/0'

dmesg > "$workspace/logs/kernel-after-hipdnn.log"
echo 0 > "$workspace/logs/hipdnn-validation.exit"
