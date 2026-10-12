#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail
source_dir="${1:?usage: build_expanded.sh <source-directory> <build-directory>}"
build_dir="${2:?missing build directory}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BC250_BUILD_JOBS:-16}"
export CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}"
export OMP_NUM_THREADS=4
cache_dir="${BC250_CCACHE_DIR:-/workspace/cache}"
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" \
    --dir "$cache_dir" --local-path "$cache_dir/data" --no-reset-stats)"
test -x "$build_dir/dist/rocm/bin/amdclang++"

# Continue the existing development build. The active release has independent
# copies of ROCm and Torch, so this cannot replace its runtime libraries.
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
    -DTHEROCK_BOOTSTRAP_CCACHE_MASQUERADE=ON \
    -DTHEROCK_AMDGPU_FAMILIES=gfx1013 \
    -DTHEROCK_TEST_AMDGPU_FAMILIES=gfx1013 \
    -DTHEROCK_ENABLE_HIPIFY=ON \
    -DTHEROCK_ENABLE_OCL_ICD=ON -DTHEROCK_ENABLE_OCL_RUNTIME=ON \
    -DTHEROCK_ENABLE_RPP=ON -DTHEROCK_ENABLE_ROCALUTION=ON \
    -DTHEROCK_RPP_ENABLE_AUDIO=ON \
    -DTHEROCK_ENABLE_HIPTENSOR=ON -DTHEROCK_HIPTENSOR_GENERIC_ONLY=ON \
    -DTHEROCK_ENABLE_HIPFILE=ON -DTHEROCK_ENABLE_HIPDNN=ON \
    -DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=ON \
    -Dhipdnn_integration_tests_CMAKE_ARGS=-DBUILD_CPP_GRAPH_TESTS=ON \
    -DTHEROCK_ENABLE_MIOPENPROVIDER=ON \
    -DTHEROCK_ENABLE_HIPKERNELPROVIDER=ON \
    -DTHEROCK_ENABLE_HIPBLASLTPROVIDER=OFF \
    -DTHEROCK_ENABLE_HIPDNN_SAMPLES=ON \
    -DTHEROCK_MIOPEN_USE_COMPOSABLE_KERNEL=OFF \
    -DTHEROCK_BUILD_TESTING=ON \
    -DTHEROCK_BACKGROUND_BUILD_JOBS=1 \
    -DrocALUTION_CMAKE_ARGS=-DUSE_HIPCXX=ON \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DFLANG_PARALLEL_COMPILE_JOBS=4 \
    '-Dhipkernelprovider_CMAKE_ARGS=-DENABLE_ASM_SDPA_ENGINE=OFF;-DENABLE_HIP_FLASH2_ENGINE=OFF;-DENABLE_HIP_MLOPS_ENGINE=ON;-DHIPKERNELPROVIDER_ENABLE_ROCKE=OFF;-DHIPDNN_ENABLE_KERNEL_INGESTOR=OFF'

for component in ocl-icd ocl-clr hipify rpp rocALUTION hipfile hipDNN \
    hipdnn_integration_tests miopenprovider hipkernelprovider hipDNN_samples; do
    echo "Building expanded component: $component"
    if [[ "$component" == rocALUTION ]]; then
        CMAKE_BUILD_PARALLEL_LEVEL=8 cmake --build "$build_dir" --target "$component"
    else
        cmake --build "$build_dir" --target "$component"
    fi
done
CMAKE_BUILD_PARALLEL_LEVEL="${BC250_HIPTENSOR_JOBS:-12}" cmake --build "$build_dir" --target hipTensor
cmake --build "$build_dir" --target dist
ccache --show-stats
