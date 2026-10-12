# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(_ffts_original_source "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rpp/third_party/ffts")
file(SHA256
  "${THEROCK_SOURCE_DIR}/patches/third-party/ffts/0001-use-supported-Clang-instruction-cache-builtin.patch"
  _ffts_patch_sha
)
set(_ffts_source "${CMAKE_CURRENT_BINARY_DIR}/source-${_ffts_patch_sha}")
therock_subproject_fetch(therock-ffts-sources
  CMAKE_PROJECT
  SOURCE_DIR "${_ffts_source}"
  DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_ffts_original_source}" "${_ffts_source}"
  PATCH_COMMAND "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/patch_third_party_source.py" "${THEROCK_SOURCE_DIR}/patches/third-party/ffts"
)
therock_cmake_subproject_declare(therock-ffts
  EXTERNAL_SOURCE_DIR "${_ffts_source}"
  FPRINT_SOURCE_DIR "${_ffts_original_source}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  COMPILER_TOOLCHAIN
    amd-llvm
  CMAKE_ARGS
    -DENABLE_SHARED=OFF
    -DENABLE_STATIC=ON
    -DDISABLE_DYNAMIC_CODE=ON
    -DGENERATE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)
therock_cmake_subproject_provide_package(therock-ffts FFTS lib/cmake/FFTS)
therock_cmake_subproject_activate(therock-ffts)
add_dependencies(therock-third-party therock-ffts)
