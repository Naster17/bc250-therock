# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(roctx VERSION 4.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

if(NOT EXISTS "${ROCTRACER_SOURCE_DIR}/src/roctx/roctx.cpp")
  message(FATAL_ERROR "ROCTRACER_SOURCE_DIR must point to the pinned ROCtracer source")
endif()

# Build the upstream marker implementation and preserve its public ABI.
add_library(roctx SHARED "${ROCTRACER_SOURCE_DIR}/src/roctx/roctx.cpp")
target_compile_features(roctx PRIVATE cxx_std_11)
target_include_directories(roctx PRIVATE
  "${ROCTRACER_SOURCE_DIR}/inc"
  "${ROCTRACER_SOURCE_DIR}/src"
)
target_link_options(roctx PRIVATE
  "-Wl,--version-script=${ROCTRACER_SOURCE_DIR}/src/roctx/exportmap"
  "-Wl,--no-undefined"
)
set_target_properties(roctx PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  OUTPUT_NAME roctx64
  DEFINE_SYMBOL ROCTX_EXPORTS
  LINK_DEPENDS "${ROCTRACER_SOURCE_DIR}/src/roctx/exportmap"
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)
install(TARGETS roctx LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${ROCTRACER_SOURCE_DIR}/inc/roctx.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/roctracer"
)
