#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Resume the upstream checksum-verified dataset download independently from
# library compilation. The converter has its own directory to avoid build races.
set -euo pipefail
source_dir="${1:?usage: fetch_sparse_matrices.sh <therock-source> <build-directory>}"
build_dir="${2:?missing build directory}"
component_source="$source_dir/rocm-libraries/projects/rocsparse"
matrix_dir="$build_dir/math-libs/BLAS/rocSPARSE/build/clients/matrices"
converter_dir="$build_dir/bc250-sparse-matrix-converter"
for attempt in 1 2 3; do
    if cmake \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
        -DPROJECT_BINARY_DIR="$converter_dir" \
        -DCMAKE_MATRICES_DIR="$matrix_dir" \
        -DCONVERT_SOURCE="$component_source/deps/convert.cpp" \
        -DROCM_PATH="$build_dir/dist/rocm" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -P "$component_source/cmake/ClientMatrices.cmake"; then
        echo SPARSE_TEST_MATRICES_VERIFIED
        exit 0
    fi
    echo "Dataset download attempt $attempt failed"
    sleep 2
done
exit 1
