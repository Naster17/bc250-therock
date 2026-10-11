// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check_hip(hipError_t error, const char *operation) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(error));
    std::exit(1);
  }
}

#define HIP_CHECK(operation) check_hip((operation), #operation)

__global__ void vector_add(const float *a, const float *b, float *output,
                           std::size_t count) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    output[i] = a[i] + b[i];
  }
}

__global__ void lds_and_atomic(unsigned *output) {
  __shared__ unsigned values[256];
  values[threadIdx.x] = threadIdx.x + 1;
  __syncthreads();
  for (unsigned stride = 128; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      values[threadIdx.x] += values[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, values[0]);
  }
}

int main() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count != 1) {
    std::fprintf(stderr, "Expected one GPU, got %d\n", device_count);
    return 1;
  }
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  std::printf("GPU=%s arch=%s warp=%d memory=%zu\n", properties.name,
              properties.gcnArchName, properties.warpSize,
              properties.totalGlobalMem);
  if (std::strncmp(properties.gcnArchName, "gfx1013", 7) != 0) {
    std::fprintf(stderr, "Expected native gfx1013\n");
    return 1;
  }

  // Include an incomplete final block and values exactly representable in FP32.
  constexpr std::size_t count = 65539;
  std::vector<float> a(count), b(count), result(count);
  for (std::size_t i = 0; i < count; ++i) {
    a[i] = static_cast<int>(i % 127) - 63;
    b[i] = static_cast<int>(i % 31) - 15;
  }
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  for (unsigned iteration = 0; iteration < 256; ++iteration) {
    float *device_a = nullptr, *device_b = nullptr, *device_output = nullptr;
    HIP_CHECK(hipMalloc(&device_a, count * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_b, count * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_output, count * sizeof(float)));
    HIP_CHECK(hipMemcpyAsync(device_a, a.data(), count * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(device_b, b.data(), count * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    hipLaunchKernelGGL(vector_add, dim3((count + 255) / 256), dim3(256), 0,
                       stream, device_a, device_b, device_output, count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyAsync(result.data(), device_output,
                             count * sizeof(float), hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (std::size_t i = 0; i < count; ++i) {
      if (result[i] != a[i] + b[i]) {
        std::fprintf(stderr, "Vector mismatch at iteration=%u index=%zu\n",
                     iteration, i);
        return 1;
      }
    }
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_a));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  std::puts("ALLOC_DISPATCH_CHURN_PASS cycles=256");

  unsigned *device_sum = nullptr;
  unsigned sum = 0;
  HIP_CHECK(hipMalloc(&device_sum, sizeof(sum)));
  HIP_CHECK(hipMemset(device_sum, 0, sizeof(sum)));
  hipLaunchKernelGGL(lds_and_atomic, dim3(17), dim3(256), 0, nullptr,
                     device_sum);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(&sum, device_sum, sizeof(sum), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device_sum));
  if (sum != 17 * (256 * 257 / 2)) {
    std::fprintf(stderr, "LDS/atomic mismatch: %u\n", sum);
    return 1;
  }
  std::puts("LDS_ATOMIC_PASS");

  for (std::size_t bytes = 4096; bytes <= 16 * 1024 * 1024; bytes *= 4) {
    std::vector<std::uint8_t> input(bytes), output(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
      input[i] = static_cast<std::uint8_t>((i * 13 + i / 256) % 251);
    }
    void *device_data = nullptr;
    HIP_CHECK(hipMalloc(&device_data, bytes));
    HIP_CHECK(
        hipMemcpy(device_data, input.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(output.data(), device_data, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(device_data));
    if (input != output) {
      std::fprintf(stderr, "Copy mismatch: bytes=%zu\n", bytes);
      return 1;
    }
    std::printf("COPY_PASS bytes=%zu\n", bytes);
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::puts("HIP_SMOKE_PASS");
}
