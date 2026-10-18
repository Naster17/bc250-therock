// SPDX-License-Identifier: MIT
#include "primitives_n216_common.hpp"
#include <cstdint>
#include <numeric>
#if N217_LIBRARY == 0
#include <rocprim/device/device_merge_sort.hpp>
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/device/device_select.hpp>
constexpr const char *library = "rocPRIM";
#elif N217_LIBRARY == 1
#include <hipcub/device/device_merge_sort.hpp>
#include <hipcub/device/device_radix_sort.hpp>
#include <hipcub/device/device_select.hpp>
constexpr const char *library = "hipCUB";
#else
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/system/hip/execution_policy.h>
constexpr const char *library = "rocThrust";
#endif

template <class T>
using UInt = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
template <class T> __host__ __device__ UInt<T> bits(T value) {
  return __builtin_bit_cast(UInt<T>, value);
}
template <class T> __host__ __device__ T from_bits(UInt<T> value) {
  return __builtin_bit_cast(T, value);
}
template <class T> __host__ __device__ constexpr UInt<T> sign_bit() {
  return UInt<T>(1) << (8 * sizeof(T) - 1);
}
template <class T> __host__ __device__ constexpr UInt<T> inf_bits() {
  if constexpr (sizeof(T) == 4)
    return 0x7f800000U;
  else
    return 0x7ff0000000000000ULL;
}
// Strict weak order: equivalent signed zeros; either pinned radix bit order
// or numeric order with every NaN equivalent and last in both directions.
struct Order {
  bool descending, nan_last;
  template <class T> __host__ __device__ bool operator()(T a, T b) const {
    auto ua = bits(a), ub = bits(b);
    const auto sign = UInt<T>(1) << (8 * sizeof(T) - 1);
    if (nan_last) {
      bool na = (ua & ~sign) > inf_bits<T>(), nb = (ub & ~sign) > inf_bits<T>();
      if (na || nb)
        return na ? false : true;
    }
    auto ra = (ua & ~sign) == 0 ? sign : (ua & sign) ? ~ua : (ua ^ sign);
    auto rb = (ub & ~sign) == 0 ? sign : (ub & sign) ? ~ub : (ub ^ sign);
    return descending ? ra > rb : ra < rb;
  }
};
struct Pick {
  int mode;
  template <class T> __host__ __device__ bool operator()(T value) const {
    auto u = bits(value), sign = UInt<T>(1) << (8 * sizeof(T) - 1);
    if (mode == 0)
      return (u & ~sign) < inf_bits<T>() && (!(u & sign) || (u & ~sign) == 0);
    if (mode == 1)
      return bool(u & sign);
    return mode == 2;
  }
};
struct FlagTrue {
  __host__ __device__ bool operator()(unsigned char flag) const {
    return flag != 0;
  }
};

template <class T>
bool cpu_before(T a, T b, bool descending, bool raw_nan_order) {
  if (!raw_nan_order) {
    if (std::isnan(a) || std::isnan(b))
      return !std::isnan(a) && std::isnan(b);
    return descending ? a > b : a < b;
  }
  // Independent scalar host encoding, matching the pinned radix contract.
  UInt<T> ua{}, ub{};
  std::memcpy(&ua, &a, sizeof(T));
  std::memcpy(&ub, &b, sizeof(T));
  const UInt<T> sign = sign_bit<T>();
  if ((ua & ~sign) == 0)
    ua = 0;
  if ((ub & ~sign) == 0)
    ub = 0;
  UInt<T> ra = (ua & sign) ? UInt<T>(~ua) : UInt<T>(ua + sign);
  UInt<T> rb = (ub & sign) ? UInt<T>(~ub) : UInt<T>(ub + sign);
  return descending ? ra > rb : ra < rb;
}
template <class T> std::vector<T> input(size_t n, int pattern) {
  std::vector<T> values(n);
  const auto sign = sign_bit<T>(), inf = inf_bits<T>();
  const UInt<T> qnan = inf | (UInt<T>(1) << (sizeof(T) == 4 ? 22 : 51));
  std::vector<T> pool{T(0),
                      from_bits<T>(sign),
                      from_bits<T>(1),
                      from_bits<T>(sign | 1),
                      from_bits<T>(inf - 1),
                      from_bits<T>(sign | (inf - 1)),
                      from_bits<T>(inf),
                      from_bits<T>(sign | inf),
                      T(1),
                      T(-1),
                      T(0),
                      from_bits<T>(sign)};
  if (pattern == 2) {
    for (UInt<T> payload : {UInt<T>(1), UInt<T>(17), UInt<T>(1)}) {
      pool.push_back(from_bits<T>(qnan | payload));
      pool.push_back(from_bits<T>(sign | qnan | payload));
    }
  }
  for (size_t i = 0; i < n; ++i) {
    if (pattern == 0) {
      values[i] = T((int(((i / 3) * 37 + 11) % 97) - 48) / 8.0);
      if (values[i] == T(0) && i % 2)
        values[i] = from_bits<T>(sign);
    } else
      values[i] = pool[(i * 7 + i / 13) % pool.size()];
  }
  return values;
}
template <class T>
void verify_bits(const Buffer<T> &out, const std::vector<T> &expected,
                 hipStream_t stream, const std::string &name) {
  auto actual = out.read(expected.size(), stream);
  require(expected.empty() || !std::memcmp(actual.data(), expected.data(),
                                           expected.size() * sizeof(T)),
          context + "/" + name + " bitwise mismatch");
}
template <class T>
void verify_pairs(const Buffer<T> &key, const Buffer<int> &value,
                  const std::vector<T> &source, bool descending, bool raw_order,
                  hipStream_t stream, const char *name) {
  std::vector<int> expected(source.size());
  std::iota(expected.begin(), expected.end(), 0);
  std::stable_sort(expected.begin(), expected.end(), [&](int a, int b) {
    return cpu_before(source[a], source[b], descending, raw_order);
  });
  std::vector<T> sorted;
  for (int i : expected)
    sorted.push_back(source[i]);
  verify_bits(key, sorted, stream, name);
  auto actual = value.read(expected.size(), stream);
  require(actual == expected,
          context + "/" + name + " stable payload/index mismatch");
  passed(name, source.size());
}
template <class T>
void radix(const Buffer<T> &in, Buffer<T> &out, const Buffer<int> &ids,
           Buffer<int> &values, bool descending, int pattern,
           hipStream_t stream) {
#if N217_LIBRARY == 0
  temporary(
      [&](void *p, size_t &b) {
        if (descending)
          return rocprim::radix_sort_pairs_desc(p, b, in.data, out.data,
                                                ids.data, values.data, in.n, 0,
                                                8 * sizeof(T), stream);
        return rocprim::radix_sort_pairs(p, b, in.data, out.data, ids.data,
                                         values.data, in.n, 0, 8 * sizeof(T),
                                         stream);
      },
      stream);
#elif N217_LIBRARY == 1
  temporary(
      [&](void *p, size_t &b) {
        if (descending)
          return hipcub::DeviceRadixSort::SortPairsDescending(
              p, b, in.data, out.data, ids.data, values.data, in.n, 0,
              8 * sizeof(T), stream);
        return hipcub::DeviceRadixSort::SortPairs(p, b, in.data, out.data,
                                                  ids.data, values.data, in.n,
                                                  0, 8 * sizeof(T), stream);
      },
      stream);
#else
  if (in.n) {
    check(hipMemcpyAsync(out.data, in.data, in.n * sizeof(T),
                         hipMemcpyDeviceToDevice, stream));
    check(hipMemcpyAsync(values.data, ids.data, in.n * sizeof(int),
                         hipMemcpyDeviceToDevice, stream));
  }
  auto first = thrust::device_pointer_cast(out.data);
  auto payload = thrust::device_pointer_cast(values.data);
  if (pattern < 2) {
    if (descending)
      thrust::stable_sort_by_key(thrust::hip::par_nosync.on(stream), first,
                                 first + in.n, payload, thrust::greater<T>());
    else
      thrust::stable_sort_by_key(thrust::hip::par_nosync.on(stream), first,
                                 first + in.n, payload, thrust::less<T>());
  } else
    thrust::stable_sort_by_key(thrust::hip::par_nosync.on(stream), first,
                               first + in.n, payload, Order{descending, false});
#endif
}
template <class T>
void comparison_sort(Buffer<T> &keys, Buffer<int> &values, bool descending,
                     hipStream_t stream) {
  Order compare{descending, true};
#if N217_LIBRARY == 0
  temporary(
      [&](void *p, size_t &b) {
        return rocprim::merge_sort(p, b, keys.data, keys.data, values.data,
                                   values.data, keys.n, compare, stream);
      },
      stream);
#elif N217_LIBRARY == 1
  temporary(
      [&](void *p, size_t &b) {
        return hipcub::DeviceMergeSort::StableSortPairs(
            p, b, keys.data, values.data, keys.n, compare, stream);
      },
      stream);
#else
  auto first = thrust::device_pointer_cast(keys.data);
  thrust::stable_sort_by_key(thrust::hip::par_nosync.on(stream), first,
                             first + keys.n,
                             thrust::device_pointer_cast(values.data), compare);
#endif
}
template <class T>
size_t select(const Buffer<T> &in, Buffer<T> &out,
              const Buffer<unsigned char> &flags, int mode,
              hipStream_t stream) {
#if N217_LIBRARY == 0 || N217_LIBRARY == 1
  Buffer<int> count(1);
  temporary(
      [&](void *p, size_t &b) {
#if N217_LIBRARY == 0
        if (mode == 4)
          return rocprim::select(p, b, in.data, flags.data, out.data,
                                 count.data, in.n, stream);
        return rocprim::select(p, b, in.data, out.data, count.data, in.n,
                               Pick{mode}, stream);
#else
        if (mode == 4)
          return hipcub::DeviceSelect::Flagged(
              p, b, in.data, flags.data, out.data, count.data, in.n, stream);
        return hipcub::DeviceSelect::If(p, b, in.data, out.data, count.data,
                                        in.n, Pick{mode}, stream);
#endif
      },
      stream);
  return size_t(count.read(1, stream)[0]);
#else
  auto first = thrust::device_pointer_cast(in.data),
       output = thrust::device_pointer_cast(out.data);
  if (mode == 4)
    return thrust::copy_if(
               thrust::hip::par_nosync.on(stream), first, first + in.n,
               thrust::device_pointer_cast(flags.data), output, FlagTrue()) -
           output;
  return thrust::copy_if(thrust::hip::par_nosync.on(stream), first,
                         first + in.n, output, Pick{mode}) -
         output;
#endif
}
template <class T> void run(size_t n, int pattern, hipStream_t stream) {
  auto source = input<T>(n, pattern);
  std::vector<int> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::vector<unsigned char> flags(n);
  for (size_t i = 0; i < n; ++i)
    flags[i] = i % 4 == 0 ? 0 : i % 4 == 1 ? 1 : i % 4 == 2 ? 2 : 255;
  Buffer<T> in(source, stream);
  Buffer<int> indices(ids, stream);
  Buffer<unsigned char> stencil(flags, stream);
  for (bool descending : {false, true}) {
    Buffer<T> output(n);
    Buffer<int> values(n);
    radix(in, output, indices, values, descending, pattern, stream);
    verify_pairs(output, values, source, descending, pattern == 2, stream,
                 descending ? "radix_or_default_desc" : "radix_or_default_asc");
    Buffer<T> inplace(source, stream);
    Buffer<int> payload(ids, stream);
    comparison_sort(inplace, payload, descending, stream);
    verify_pairs(inplace, payload, source, descending, false, stream,
                 descending ? "nan_last_stable_desc" : "nan_last_stable_asc");
  }
  for (int mode = 0; mode < 5; ++mode) {
    Buffer<T> output(n);
    const size_t count = select(in, output, stencil, mode, stream);
    std::vector<T> expected;
    for (size_t i = 0; i < n; ++i) {
      bool pick = mode == 0   ? std::isfinite(source[i]) && source[i] >= T(0)
                  : mode == 1 ? std::signbit(source[i])
                  : mode == 2 ? true
                  : mode == 3 ? false
                              : flags[i] != 0;
      if (pick)
        expected.push_back(source[i]);
    }
    require(count == expected.size(), context + " selected count mismatch");
    verify_bits(output, expected, stream, "stable selection");
    passed("select_mode" + std::to_string(mode), count);
  }
  in.unchanged(source, stream);
  indices.unchanged(ids, stream);
  stencil.unchanged(flags, stream);
}
template <class T> void partial_bits_refusal(hipStream_t stream) {
#if N217_LIBRARY != 2
  auto source = input<T>(33, 0);
  Buffer<T> in(source, stream), out(33);
  std::vector<int> ids(33);
  std::iota(ids.begin(), ids.end(), 0);
  Buffer<int> values(ids, stream), result(33);
  for (auto range : {std::pair<unsigned, unsigned>{1, 8 * sizeof(T)},
                     {0, 8 * sizeof(T) - 1}}) {
    size_t bytes = 0;
#if N217_LIBRARY == 0
    auto status = rocprim::radix_sort_pairs(nullptr, bytes, in.data, out.data,
                                            values.data, result.data, 33,
                                            range.first, range.second, stream);
#else
    auto status = hipcub::DeviceRadixSort::SortPairs(
        nullptr, bytes, in.data, out.data, values.data, result.data, 33,
        range.first, range.second, stream);
#endif
    require(status == hipErrorInvalidValue,
            "floating partial bit range must be rejected");
    out.read(0, stream);
    result.read(0, stream);
    passed("refuse_partial_bits_" + std::to_string(range.first) + "_" +
               std::to_string(range.second),
           0);
  }
  in.unchanged(source, stream);
  values.unchanged(ids, stream);
#endif
}
int main() {
  try {
    native_device();
    Stream private_stream;
    for (int s = 0; s < 2; ++s) {
      hipStream_t stream = s ? private_stream.value : nullptr;
      for (size_t n : {size_t(0), size_t(1), size_t(33), size_t(257),
                       size_t(4099), size_t(65537)}) {
        for (int pattern = 0; pattern < 3; ++pattern) {
          context = "f32/n" + std::to_string(n) + "/pattern" +
                    std::to_string(pattern) + "/stream" + std::to_string(s);
          run<float>(n, pattern, stream);
          context.replace(0, 3, "f64");
          run<double>(n, pattern, stream);
        }
      }
      context = "f32/contracts/stream" + std::to_string(s);
      partial_bits_refusal<float>(stream);
      context.replace(0, 3, "f64");
      partial_bits_refusal<double>(stream);
    }
    std::printf("{\"status\":\"passed\",\"architecture\":\"gfx1013\","
                "\"library\":\"%s\",\"cases\":%zu,\"streams\":2,\"guards\":"
                "true,\"max_workspace_bytes\":%zu,\"checks\":[",
                library, rows.size(), max_workspace);
    for (size_t i = 0; i < rows.size(); ++i)
      std::printf("%s%s", i ? "," : "", rows[i].c_str());
    std::puts("]}");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s: %s\n", library, error.what());
    return 1;
  }
}
