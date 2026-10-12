// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(expression)                                                 \
  do {                                                                         \
    cudaError_t status = (expression);                                         \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s: %s\n", #expression,                            \
                   cudaGetErrorString(status));                                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void transform(int *values, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] = values[index] * 3 + 7;
}

int main() {
  constexpr int count = 1027;
  std::vector<int> values(count);
  for (int i = 0; i < count; ++i)
    values[i] = i;
  int *device = nullptr;
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(
      cudaMalloc(reinterpret_cast<void **>(&device), count * sizeof(int)));
  CHECK_CUDA(cudaMemcpyAsync(device, values.data(), count * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  transform<<<(count + 255) / 256, 256, 0, stream>>>(device, count);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpyAsync(values.data(), device, count * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  for (int i = 0; i < count; ++i) {
    if (values[i] != i * 3 + 7) {
      std::fprintf(stderr, "Mismatch at %d: %d\n", i, values[i]);
      return 1;
    }
  }
  CHECK_CUDA(cudaFree(device));
  CHECK_CUDA(cudaStreamDestroy(stream));
  std::puts("HIPIFY_TRANSLATED_KERNEL_PASS");
}
