#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run GPU checks serially only after the developer distribution is complete.
set -eu
workspace="${1:?usage: finish_expanded_validation_on_host.sh <workspace-directory>}"
test "$(id -u)" -eq 0
echo 1 > "$workspace/logs/expanded-validation.exit"
while ! test -f "$workspace/logs/expanded-build.exit"; do
    sleep 15
done
test "$(cat "$workspace/logs/expanded-build.exit")" = 0
status=0
for component in rpp rpp_audio opencl openmp hipfile rocalution_diagnostic \
    hiptensor hipdnn profiler rocprim_arch; do
    echo "Validating expanded component: $component"
    if sh "$workspace/helpers/validate_${component}_on_host.sh" "$workspace"; then
        echo "Expanded component gate passed: $component"
    else
        echo "Expanded component gate failed: $component"
        status=1
    fi
done
echo "$status" > "$workspace/logs/expanded-validation.exit"
exit "$status"
