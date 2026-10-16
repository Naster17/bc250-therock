// SPDX-License-Identifier: MIT
// Compile-only contract for the bundled legacy CK native gfx1013 profile.
#include "utility/config.hpp"

#ifndef CK_AMD_GPU_GFX1013
#error A02 requires the native gfx1013 target macro.
#endif
static_assert(CK_BUFFER_RESOURCE_3RD_DWORD == 0x31014000);
#if CK_USE_AMD_INLINE_ASM || CK_USE_AMD_INNER_PRODUCT_INLINE_ASM ||            \
    CK_USE_AMD_XDLOPS
#error A02 native profile must use portable scalar arithmetic.
#endif
#if defined(CK_USE_AMD_V_DOT2_F32_F16) || defined(CK_USE_AMD_V_DOT4_I32_I8)
#error Native gfx1013 must not advertise packed-dot instructions.
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1013__)
#error A02 must compile for native gfx1013.
#endif

__global__ void native_profile(int *output) { output[0] = warpSize; }
