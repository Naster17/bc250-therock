# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_cmake_subproject_declare(therock-ffts
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rpp/third_party/ffts"
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
