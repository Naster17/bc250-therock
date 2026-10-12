#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -eu
workspace="${1:?usage: validate_hiptensor_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
test "$(cat "$workspace/logs/expanded-build.exit")" = 0
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
image=localhost/bc250-rocm-build:10.2-ccache
prefix=/workspace/build/dist/rocm
dmesg > "$workspace/logs/kernel-before-hiptensor.log"

run_tests() {
    binary="$1"
    config="$2"
    set +e
    podman run --rm --network none --device /dev/kfd --device /dev/dri/renderD128 \
        -v "$workspace/build:/workspace/build:ro" \
        -v "$workspace/logs:/workspace/logs:rw" \
        -w "$prefix/bin/hiptensor" \
        -e HSA_ENABLE_SDMA=0 \
        -e LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib" \
        -e GTEST_OUTPUT="xml:/workspace/logs/hiptensor-$binary.xml" \
        "$image" timeout --kill-after=10s 300s "../$binary" \
        -y "configs/validation/quick/$config" -v ON --hot_runs 1 --cold_runs 0 \
        > "$workspace/logs/hiptensor-$binary.log" 2>&1
    status=$?
    set -e
    echo "$status" > "$workspace/logs/hiptensor-$binary.exit"
    test "$status" -eq 0
    python3 - "$workspace/logs/hiptensor-$binary.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
assert int(root.attrib["tests"]) > 0, root.attrib
for field in ("failures", "disabled", "errors"):
    assert int(root.attrib.get(field, 0)) == 0, root.attrib
for case in root.iter("testcase"):
    assert case.attrib.get("result") == "completed", case.attrib
    assert case.find("skipped") is None, case.attrib
print("HIPTENSOR_GENERIC_GPU_TESTS_PASS", root.attrib["tests"])
PY
}

run_tests rank2_reduction_test 03_reduction/rank2_test_params.yaml
run_tests rank2_elementwise_permute_test 02_elementwise/permutation/rank2_test_params.yaml
run_tests rank2_elementwise_binary_op_test 02_elementwise/binary_op/rank2_binary_op_test_params.yaml
run_tests rank2_elementwise_trinary_op_test 02_elementwise/trinary_op/rank2_trinary_op_test_params.yaml

dmesg > "$workspace/logs/kernel-after-hiptensor.log"
echo 0 > "$workspace/logs/hiptensor-validation.exit"
