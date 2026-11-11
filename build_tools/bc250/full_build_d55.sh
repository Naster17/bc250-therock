#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
profile=$1
cd /workspace/pytorch-work
recipe=/workspace/recipes/build-fresh-${profile}.sh
# Use the archived environment up to the pip command; the offline pip
# invocation below replaces the recipe's networked one.
source <(sed '/^python -m pip install /,$d' "$recipe")
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export CCACHE_READONLY=1 CCACHE_NOSTATS=1
export CCACHE_TEMPDIR=/tmp/ccache CCACHE_LOGFILE=/tmp/ccache.log
export CMAKE_TOOLCHAIN_FILE=/probe/offline_inputs_d54.cmake
save_diagnostics() {
    if [[ -f build/CMakeFiles/CMakeConfigureLog.yaml ]]; then
        cp build/CMakeFiles/CMakeConfigureLog.yaml /evidence/
    fi
    if [[ -f build/CMakeCache.txt ]]; then
        cp build/CMakeCache.txt /evidence/
    fi
    for metric in memory.max memory.peak memory.swap.max memory.events cpu.max cpuset.cpus.effective; do
        if [[ -r /sys/fs/cgroup/$metric ]]; then
            printf '%s: ' "$metric"
            cat "/sys/fs/cgroup/$metric"
        fi
    done > /evidence/cgroup.txt
    while IFS= read -r path; do
        if [[ -f $path ]]; then
            sha256sum "$path"
        fi
    done < /probe/hip-paths.txt > /evidence/hip-files.sha256
    if [[ -f /tmp/ccache.log ]]; then
        grep -c 'Result: .*cache_hit' /tmp/ccache.log > /evidence/ccache-hits.txt || true
        grep -c 'Result: cache_miss' /tmp/ccache.log > /evidence/ccache-misses.txt || true
    fi
}
trap save_diagnostics EXIT
[[ ! -e /evidence/full-build-passed ]]
python -m pip install -e . -v --no-build-isolation --no-deps
target=build/caffe2/CMakeFiles/torch_hip.dir/__/aten/src/ATen/native/hip/BatchLinearAlgebraEig.hip.o
"$HIP_CLANG_PATH/llvm-objdump" --offloading "$target" > /evidence/offload.txt
grep -qw gfx1013 /evidence/offload.txt
sha256sum "$target" > /evidence/objects.sha256
sha256sum torch/lib/*.so torch/_C*.so > /evidence/libs.sha256
printf 'Full clean build completed\n' > /evidence/full-build-passed
