#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
profile=$1
cd /workspace/pytorch-work
target=caffe2/CMakeFiles/torch_hip.dir/__/aten/src/ATen/native/hip/BatchLinearAlgebraEig.hip.o
rccl_target=caffe2/CMakeFiles/torch_hip.dir/__/torch/csrc/distributed/c10d/NCCLUtils.cpp.o
if [[ ${2:-} == native-step ]]; then
    targets=("$target")
    if [[ $profile == rccl-single ]]; then
        targets+=("$rccl_target")
    fi
    cp build/CMakeCache.txt /evidence/
    nnpack=confu-deps/NNPACK/src/x86_64-fma/2d-fourier-8x8.py.o
    ninja -C build -j 4 "${targets[@]}" "$nnpack"
    ninja -C build -t deps "${targets[@]}" > /evidence/target-deps.txt
    sha256sum "${targets[@]/#/build/}" "build/$nnpack" > /evidence/objects.sha256
    "$HIP_CLANG_PATH/llvm-objdump" --offloading "build/$target" > /evidence/offload.txt
    grep -qw gfx1013 /evidence/offload.txt
    libraries=(-lamdhip64 -lMIOpen -lhipblas -lhipfft -lhipsparse -lhipsolver -lrocprofiler-sdk)
    if [[ $profile == rccl-single ]]; then
        libraries+=(-L/workspace/rccl/lib -lrccl)
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/rccl/lib"
    fi
    printf 'int main() { return 0; }\n' | g++ -x c++ - -o /tmp/rocm-link \
        -Wl,--no-as-needed -L"$ROCM_PATH/lib" "${libraries[@]}"
    readelf -d /tmp/rocm-link > /evidence/link-dynamic.txt
    ldd /tmp/rocm-link > /evidence/link-loader.txt
    if grep -q 'not found' /evidence/link-loader.txt; then
        cat /evidence/link-loader.txt >&2
        exit 1
    fi
    /tmp/rocm-link
    printf 'Native targets compiled successfully\n' > /evidence/target-passed
    # The editable backend cannot produce a wheel from a targeted build.
    exit 54
fi

recipe=/workspace/recipes/build-fresh-${profile}.sh
# Use the archived environment up to the pip command. The wrapper supplies
# a bounded native step and leaves the archived recipe untouched.
source <(sed '/^python -m pip install /,$d' "$recipe")
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export CCACHE_READONLY=1 CCACHE_NOSTATS=1
export CCACHE_TEMPDIR=/tmp/ccache
export CMAKE_TOOLCHAIN_FILE=/probe/offline_inputs_d54.cmake
save_diagnostics() {
    if [[ -f build/CMakeFiles/CMakeConfigureLog.yaml ]]; then
        cp build/CMakeFiles/CMakeConfigureLog.yaml /evidence/
    fi
    for metric in memory.max memory.peak memory.swap.max memory.events cpu.max cpuset.cpus.effective; do
        if [[ -r /sys/fs/cgroup/$metric ]]; then
            printf '%s: ' "$metric"
            cat "/sys/fs/cgroup/$metric"
        fi
    done > /evidence/cgroup.txt
    while IFS= read -r path; do
        sha256sum "$path"
    done < /probe/hip-paths.txt > /evidence/hip-files.sha256
}
trap save_diagnostics EXIT
export BUILD_CUSTOM_STEP="bash /probe/probe_clean_build_d54.sh $profile native-step"
[[ ! -e /evidence/target-passed ]]
set +e
python -m pip install -e . -v --no-build-isolation --no-deps
rc=$?
set -e
[[ $rc == 1 && -s /evidence/target-passed ]]
