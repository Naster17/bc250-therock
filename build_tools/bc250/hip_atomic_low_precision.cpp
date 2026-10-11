// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(operation)                                                   \
  do {                                                                         \
    const auto status = (operation);                                           \
    if (status != hipSuccess) {                                                \
      std::fprintf(stderr, "%s: %s\n", #operation, hipGetErrorString(status)); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

__global__ void add_half(__half *values, float *old_values) {
  unsafeAtomicAdd(reinterpret_cast<__half2 *>(values),
                  __floats2half2_rn(1.0f, 2.0f));
  unsafeAtomicAdd(values + 2, __float2half(1.0f));
  unsafeAtomicAdd(values + 3, __float2half(2.0f));
  if (threadIdx.x == 0) {
    const auto old = unsafeAtomicAdd(reinterpret_cast<__half2 *>(values + 6),
                                     __floats2half2_rn(1.0f, 2.0f));
    old_values[0] = __half2float(__low2half(old));
    old_values[1] = __half2float(__high2half(old));
  }
}

__global__ void add_bfloat(__hip_bfloat16 *values, float *old_values) {
  unsafeAtomicAdd(
      reinterpret_cast<__hip_bfloat162 *>(values),
      __halves2bfloat162(__float2bfloat16(1.0f), __float2bfloat16(2.0f)));
  unsafeAtomicAdd(values + 2, __float2bfloat16(1.0f));
  unsafeAtomicAdd(values + 3, __float2bfloat16(2.0f));
  if (threadIdx.x == 0) {
    const auto old = unsafeAtomicAdd(
        reinterpret_cast<__hip_bfloat162 *>(values + 6),
        __halves2bfloat162(__float2bfloat16(1.0f), __float2bfloat16(2.0f)));
    old_values[2] = __bfloat162float(__low2bfloat16(old));
    old_values[3] = __bfloat162float(__high2bfloat16(old));
  }
}

int main() {
  const float initial[] = {0, 0, 0, 0, 7, 11, 7, 11};
  const float expected[] = {128, 256, 128, 256, 7, 11, 8, 13};
  __half half_values[8];
  __hip_bfloat16 bfloat_values[8];
  for (int i = 0; i < 8; ++i) {
    half_values[i] = __float2half(initial[i]);
    bfloat_values[i] = __float2bfloat16(initial[i]);
  }
  __half *device_half = nullptr;
  __hip_bfloat16 *device_bfloat = nullptr;
  float *device_old = nullptr;
  HIP_CHECK(hipMalloc(&device_half, sizeof(half_values)));
  HIP_CHECK(hipMalloc(&device_bfloat, sizeof(bfloat_values)));
  HIP_CHECK(hipMalloc(&device_old, 4 * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_half, half_values, sizeof(half_values),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_bfloat, bfloat_values, sizeof(bfloat_values),
                      hipMemcpyHostToDevice));
  hipLaunchKernelGGL(add_half, dim3(1), dim3(128), 0, nullptr, device_half,
                     device_old);
  HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(add_bfloat, dim3(1), dim3(128), 0, nullptr, device_bfloat,
                     device_old);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(half_values, device_half, sizeof(half_values),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(bfloat_values, device_bfloat, sizeof(bfloat_values),
                      hipMemcpyDeviceToHost));
  float old_values[4];
  HIP_CHECK(hipMemcpy(old_values, device_old, sizeof(old_values),
                      hipMemcpyDeviceToHost));
  bool passed = true;
  for (int i = 0; i < 8; ++i) {
    const auto half_result = __half2float(half_values[i]);
    const auto bfloat_result = __bfloat162float(bfloat_values[i]);
    if (half_result != expected[i] || bfloat_result != expected[i]) {
      std::fprintf(stderr, "lane=%d expected=%g half=%g bfloat=%g\n", i,
                   expected[i], half_result, bfloat_result);
      passed = false;
    }
  }
  for (int i = 0; i < 4; ++i) {
    if (old_values[i] != initial[6 + i % 2]) {
      std::fprintf(stderr, "old[%d]=%g expected=%g\n", i, old_values[i],
                   initial[6 + i % 2]);
      passed = false;
    }
  }
  HIP_CHECK(hipFree(device_old));
  HIP_CHECK(hipFree(device_bfloat));
  HIP_CHECK(hipFree(device_half));
  if (!passed) {
    return 1;
  }
  std::puts("FP16_BF16_ATOMIC_PASS");
}
