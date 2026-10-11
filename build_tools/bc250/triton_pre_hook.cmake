# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Triton appends its own linker pool to this property. Defining the compile pool
# here also limits the repository's direct Ninja invocation through `make`.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS "compile=$ENV{MAX_JOBS}")
