# SPDX-License-Identifier: MIT
# ORT copies Abseil include paths into some targets without forwarding SYSTEM.
# Keep strict warnings in ORT while treating this pinned third-party header tree
# consistently with ABSL_USE_SYSTEM_INCLUDES under Clang 24.
include_directories(SYSTEM "${CMAKE_BINARY_DIR}/_deps/abseil_cpp-src")
