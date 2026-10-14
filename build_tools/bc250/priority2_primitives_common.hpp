// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

inline void check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
struct Even {
  __host__ __device__ bool operator()(int x) const { return x % 2 == 0; }
};
struct Twice {
  __host__ __device__ int operator()(int x) const { return 2 * x; }
};
struct Buffer {
  static constexpr int guard = 0x13579bdf;
  int *raw{}, *data{};
  size_t n;
  Buffer(size_t count, hipStream_t stream) : n(count) {
    check(hipMalloc(&raw, (n + 2) * sizeof(int)));
    data = raw + 1;
    std::vector<int> init(n + 2, guard);
    check(hipMemcpyAsync(raw, init.data(), init.size() * sizeof(int),
                         hipMemcpyHostToDevice, stream));
    check(hipStreamSynchronize(stream));
  }
  ~Buffer() { (void)hipFree(raw); }
  Buffer(const Buffer &) = delete;
  void upload(const std::vector<int> &v, hipStream_t stream) {
    if (v.size() > n)
      throw std::runtime_error("upload exceeds allocation");
    check(hipMemcpyAsync(data, v.data(), v.size() * sizeof(int),
                         hipMemcpyHostToDevice, stream));
    check(hipStreamSynchronize(stream));
  }
  std::vector<int> read(hipStream_t stream) const {
    std::vector<int> result(n + 2);
    check(hipMemcpyAsync(result.data(), raw, result.size() * sizeof(int),
                         hipMemcpyDeviceToHost, stream));
    check(hipStreamSynchronize(stream));
    if (result.front() != guard || result.back() != guard)
      throw std::runtime_error("guard overwritten");
    return {result.begin() + 1, result.end() - 1};
  }
};
inline int cases = 0;
inline void verify(const Buffer &output, const std::vector<int> &expected,
                   hipStream_t stream, const char *name) {
  auto actual = output.read(stream);
  if (expected.size() > actual.size() ||
      !std::equal(expected.begin(), expected.end(), actual.begin()))
    throw std::runtime_error(std::string(name) + " CPU reference mismatch");
  ++cases;
}
template <class F> void temporary(F f) {
  size_t bytes = 0;
  check(f(nullptr, bytes));
  void *storage{};
  check(hipMalloc(&storage, std::max<size_t>(bytes, 1)));
  auto status = f(storage, bytes);
  auto released = hipFree(storage);
  check(status);
  check(released);
}
inline void native_device() {
  hipDeviceProp_t p{};
  check(hipGetDeviceProperties(&p, 0));
  if (std::strncmp(p.gcnArchName, "gfx1013", 7) || p.warpSize != 32)
    throw std::runtime_error("requires native gfx1013 wave32");
}
inline std::vector<int> input(size_t n) {
  std::vector<int> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = int((i * 37 + 11) % 97) - 48;
  return v;
}
inline int finish(const char *library) {
  printf("{\"status\":\"passed\",\"library\":\"%s\",\"architecture\":"
         "\"gfx1013\",\"cases\":%d,\"streams\":2,\"guards\":true}\n",
         library, cases);
  return 0;
}
