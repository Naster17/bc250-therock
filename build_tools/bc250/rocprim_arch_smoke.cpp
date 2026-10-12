// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <rocprim/device/config_types.hpp>

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

using rocprim::detail::gen;
using rocprim::detail::target_arch;

static void hip_check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}

__global__ void identify(unsigned *output) {
  constexpr auto arch = rocprim::detail::device_target_arch();
#if defined(__HIP_DEVICE_COMPILE__)
  static_assert(arch == target_arch::gfx1013);
  static_assert(rocprim::detail::gen_from_target_arch(arch) == gen::rdna1);
#endif
  output[0] = unsigned(arch);
  output[1] = unsigned(rocprim::detail::gen_from_target_arch(arch));
}

int main() {
  try {
    if (rocprim::detail::parse_gcn_arch("gfx1013") != target_arch::gfx1013 ||
        rocprim::detail::parse_gcn_arch("gfx1013:xnack-") !=
            target_arch::gfx1013 ||
        rocprim::detail::parse_gcn_arch("gfx10130") != target_arch::unknown ||
        rocprim::detail::parse_gcn_arch("gfx1013suffix") !=
            target_arch::unknown)
      throw std::runtime_error("rocPRIM architecture parsing failed");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1013", 7) ||
        properties.warpSize != 32)
      throw std::runtime_error("expected native gfx1013 wave32 device");
    hipStream_t stream;
    hip_check(hipStreamCreate(&stream));
    target_arch arch;
    for (int repeat = 0; repeat < 2; ++repeat) {
      hip_check(rocprim::detail::get_device_arch(0, arch));
      if (arch != target_arch::gfx1013)
        throw std::runtime_error(
            "rocPRIM device cache returned an unknown architecture");
    }
    for (auto selected : {hipStreamDefault, stream}) {
      hip_check(rocprim::detail::host_target_arch(selected, arch));
      if (arch != target_arch::gfx1013 ||
          rocprim::detail::gen_from_target_arch(arch) != gen::rdna1)
        throw std::runtime_error(
            "rocPRIM stream target differs from native architecture");
    }
    unsigned *device, output[2];
    hip_check(hipMalloc(&device, sizeof(output)));
    hipLaunchKernelGGL(identify, dim3(1), dim3(1), 0, stream, device);
    hip_check(hipGetLastError());
    hip_check(hipMemcpyAsync(output, device, sizeof(output),
                             hipMemcpyDeviceToHost, stream));
    hip_check(hipStreamSynchronize(stream));
    if (output[0] != unsigned(target_arch::gfx1013) ||
        output[1] != unsigned(gen::rdna1))
      throw std::runtime_error("rocPRIM compiler and runtime targets differ");
    hip_check(hipFree(device));
    hip_check(hipStreamDestroy(stream));
    puts("ROCPRIM_NATIVE_ARCH_DISPATCH_PASS");
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
