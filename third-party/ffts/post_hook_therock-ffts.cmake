# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(CMakePackageConfigHelpers)
set_target_properties(ffts_static PROPERTIES OUTPUT_NAME ffts)
configure_file(
  "${THEROCK_SOURCE_DIR}/third-party/ffts/FFTSConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/FFTSConfig.cmake"
  @ONLY
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FFTSConfigVersion.cmake"
  VERSION 0.9.0
  COMPATIBILITY AnyNewerVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/FFTSConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FFTSConfigVersion.cmake"
  DESTINATION lib/cmake/FFTS
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/COPYRIGHT"
  DESTINATION share/doc/ffts
)
