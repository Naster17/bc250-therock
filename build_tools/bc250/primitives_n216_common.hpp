// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

inline void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
inline void check(hipError_t status) {
  require(status == hipSuccess, hipGetErrorString(status));
}
struct Stream {
  hipStream_t value{};
  Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
  ~Stream() { (void)hipStreamDestroy(value); }
};
template <class T> struct Buffer {
  unsigned char *raw{};
  T *data{};
  size_t n;
  static constexpr size_t guard = 256;
  Buffer(size_t count) : n(count) {
    check(hipMalloc(&raw, n * sizeof(T) + 2 * guard));
    data = reinterpret_cast<T *>(raw + guard);
    // Synchronous initialization precedes work on nonblocking streams.
    check(hipMemset(raw, 0xa5, n * sizeof(T) + 2 * guard));
    check(hipDeviceSynchronize());
  }
  Buffer(const std::vector<T> &values, hipStream_t stream)
      : Buffer(values.size()) {
    if (n)
      check(hipMemcpyAsync(data, values.data(), n * sizeof(T),
                           hipMemcpyHostToDevice, stream));
    check(hipStreamSynchronize(stream));
  }
  Buffer(const Buffer &) = delete;
  ~Buffer() { (void)hipFree(raw); }
  std::vector<unsigned char> bytes(hipStream_t stream) const {
    std::vector<unsigned char> host(n * sizeof(T) + 2 * guard);
    check(hipMemcpyAsync(host.data(), raw, host.size(), hipMemcpyDeviceToHost,
                         stream));
    check(hipStreamSynchronize(stream));
    require(std::all_of(host.begin(), host.begin() + guard,
                        [](auto c) { return c == 0xa5; }),
            "prefix guard overwritten");
    require(std::all_of(host.end() - guard, host.end(),
                        [](auto c) { return c == 0xa5; }),
            "suffix guard overwritten");
    return host;
  }
  std::vector<T> read(size_t count, hipStream_t stream) const {
    require(count <= n, "read exceeds payload");
    auto host = bytes(stream);
    // Unused output payload is protected too (reduce-by-key count/tails).
    require(std::all_of(host.begin() + guard + count * sizeof(T),
                        host.end() - guard, [](auto c) { return c == 0xa5; }),
            "unused output tail overwritten");
    std::vector<T> result(count);
    if (count)
      std::memcpy(result.data(), host.data() + guard, count * sizeof(T));
    return result;
  }
  void unchanged(const std::vector<T> &expected, hipStream_t stream) const {
    auto host = bytes(stream);
    require(expected.size() == n &&
                (!n || !std::memcmp(expected.data(), host.data() + guard,
                                    n * sizeof(T))),
            "input overwritten");
  }
};
inline size_t max_workspace = 0;
template <class F> void temporary(F f, hipStream_t stream) {
  size_t bytes = 0;
  check(f(nullptr, bytes));
  max_workspace = std::max(max_workspace, bytes);
  Buffer<unsigned char> storage(std::max<size_t>(bytes, 1));
  check(f(storage.data, bytes));
  check(hipGetLastError());
  check(hipStreamSynchronize(stream));
  storage.bytes(stream);
}
template <class T> struct Plus {
  __host__ __device__ T operator()(T a, T b) const { return a + b; }
};
struct Equal {
  __host__ __device__ bool operator()(int a, int b) const { return a == b; }
};
template <class T> struct ArgMax {
  T value;
  int first;
};
template <class T> struct ChooseMax {
  __host__ __device__ ArgMax<T> operator()(ArgMax<T> a, ArgMax<T> b) const {
    return a.value > b.value || (a.value == b.value && a.first < b.first) ? a
                                                                          : b;
  }
};
inline std::vector<std::string> rows;
inline std::string context;
inline void passed(const std::string &name, size_t elements,
                   long double max_error = 0) {
  char buffer[512];
  std::snprintf(
      buffer, sizeof(buffer),
      "{\"name\":\"%s/"
      "%s\",\"elements\":%zu,\"max_abs_error\":%.18Lg,\"status\":\"passed\"}",
      context.c_str(), name.c_str(), elements, max_error);
  rows.emplace_back(buffer);
  std::fprintf(stderr, "PASS %s/%s\n", context.c_str(), name.c_str());
}
template <class T>
void verify(const Buffer<T> &output, const std::vector<long double> &expected,
            hipStream_t stream, const char *name, bool exact) {
  auto actual = output.read(expected.size(), stream);
  long double max_error = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const long double error =
        std::abs(static_cast<long double>(actual[i]) - expected[i]);
    const long double tolerance =
        exact ? 0
              : (std::is_same_v<T, float>
                     ? 5e-4L + 4e-5L * std::abs(expected[i])
                     : 3e-11L + 5e-12L * std::abs(expected[i]));
    require(std::isfinite(actual[i]) && error <= tolerance,
            context + "/" + name + " mismatch at " + std::to_string(i) +
                " error=" + std::to_string(double(error)));
    max_error = std::max(max_error, error);
  }
  passed(name, expected.size(), max_error);
}
template <class T>
void verify(const Buffer<ArgMax<T>> &output,
            const std::vector<ArgMax<T>> &expected, hipStream_t stream,
            const char *name) {
  auto actual = output.read(expected.size(), stream);
  for (size_t i = 0; i < expected.size(); ++i)
    require(actual[i].value == expected[i].value &&
                actual[i].first == expected[i].first,
            context + "/" + name + " argmax/tie mismatch at " +
                std::to_string(i));
  passed(name, expected.size());
}
inline std::vector<int> offsets(size_t n) {
  std::vector<int> result{0, 0};
  const size_t lengths[] = {1, 31, 32, 33, 255, 257, 4099, 0};
  size_t i = 0;
  while (size_t(result.back()) < n)
    result.push_back(
        int(std::min(n, size_t(result.back()) + lengths[i++ % 8])));
  result.push_back(int(n));
  return result;
}
inline std::vector<int> keys_from_offsets(const std::vector<int> &bounds) {
  std::vector<int> keys(bounds.back());
  int active = 0;
  for (size_t j = 0; j + 1 < bounds.size(); ++j) {
    if (bounds[j] != bounds[j + 1]) {
      std::fill(keys.begin() + bounds[j], keys.begin() + bounds[j + 1],
                active++ % 3);
    }
  }
  return keys;
}
inline void native_device() {
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, 0));
  require(std::string(properties.gcnArchName).substr(0, 7) == "gfx1013" &&
              properties.warpSize == 32,
          "native gfx1013 wave32 required");
}
