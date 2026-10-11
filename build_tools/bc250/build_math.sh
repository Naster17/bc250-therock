#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Continue the same core build with native math/ML dependencies for PyTorch.
set -euo pipefail

source_dir="${1:?usage: build_math.sh <source-directory> <build-directory>}"
build_dir="${2:?usage: build_math.sh <source-directory> <build-directory>}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cache_dir="${BC250_CCACHE_DIR:-/workspace/cache}"
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" \
    --dir "$cache_dir" --local-path "$cache_dir/data" --no-reset-stats)"

test -x "$build_dir/dist/rocm/bin/amdclang++"
# Some Tensile packaging steps write generated files alongside its Python
# package. Use a copy in the build tree while keeping the source snapshot read-only.
mkdir -p "$build_dir/tensile-source"
cp -a "$source_dir/rocm-libraries/shared/tensile/." "$build_dir/tensile-source/"

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
    -DTHEROCK_BOOTSTRAP_CCACHE_MASQUERADE=ON \
    -DTHEROCK_AMDGPU_FAMILIES=gfx1013 \
    -DTHEROCK_TEST_AMDGPU_FAMILIES=gfx1013 \
    -DTHEROCK_ENABLE_ALL=OFF \
    -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
    -DTHEROCK_ENABLE_CORE_AMDSMI=ON \
    -DTHEROCK_ENABLE_ROCTX=OFF \
    -DTHEROCK_ENABLE_BLAS=ON \
    -DTHEROCK_ENABLE_RAND=ON \
    -DTHEROCK_ENABLE_FFT=ON \
    -DTHEROCK_ENABLE_PRIM=ON \
    -DTHEROCK_ENABLE_SPARSE=ON \
    -DTHEROCK_ENABLE_SOLVER=ON \
    -DTHEROCK_ENABLE_MIOPEN=ON \
    -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON \
    -DTHEROCK_MIOPEN_USE_COMPOSABLE_KERNEL=OFF \
    -DTHEROCK_BUILD_TESTING=ON \
    -DTHEROCK_ENABLE_LLVM_TESTS=OFF \
    -DTHEROCK_BUILD_COMGR_TESTS=OFF \
    -DTHEROCK_BACKGROUND_BUILD_JOBS=1 \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DFLANG_PARALLEL_COMPILE_JOBS="${BC250_FLANG_JOBS:-4}" \
    -Damdsmi_CMAKE_ARGS=-DENABLE_ESMI_LIB=OFF \
    "-DrocBLAS_CMAKE_ARGS=-DBUILD_WITH_PIP=OFF;-DTensile_ROOT=$build_dir/tensile-source/Tensile;-DTensile_CPU_THREADS=4;-DTensile_KEEP_BUILD_TMP=OFF"

# Build the wrappers after their backend libraries. Independent Python kernel
# generators have their own worker limits, separate from Ninja's compile jobs.
for component in rocRAND hipRAND rocPRIM hipCUB rocThrust rocBLAS hipBLASLt \
    rocFFT hipFFT rocSPARSE hipSPARSE rocSOLVER hipSOLVER hipBLAS MIOpen; do
    echo "Building native component: $component"
    cmake --build "$build_dir" --target "$component"
done

# CK's template instantiations need more RAM per compiler process.
CMAKE_BUILD_PARALLEL_LEVEL="${BC250_CK_JOBS:-6}" \
    cmake --build "$build_dir" --target composable_kernel
cmake --build "$build_dir" --target dist
ccache --show-stats
