# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run the upstream HIP source generator inside the authorized pip build.
# Compiler try_compile projects do not contain this entry point.
if(EXISTS "${CMAKE_SOURCE_DIR}/tools/amd_build/build_amd.py")
  get_property(_bc250_hipified GLOBAL PROPERTY BC250_PYTORCH_HIPIFIED)
  if(NOT _bc250_hipified)
    find_program(_bc250_python NAMES python REQUIRED)
    execute_process(
      COMMAND "${_bc250_python}" "${CMAKE_SOURCE_DIR}/tools/amd_build/build_amd.py"
        --project-directory "${CMAKE_SOURCE_DIR}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      COMMAND_ERROR_IS_FATAL ANY
    )
    set_property(GLOBAL PROPERTY BC250_PYTORCH_HIPIFIED TRUE)
  endif()
endif()

# HIP package imports must also use the native target in a GPU-less container.
set(GPU_TARGETS gfx1013 CACHE STRING "Native BC-250 target")
set(AMDGPU_TARGETS gfx1013 CACHE STRING "Native BC-250 target")
