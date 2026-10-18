// SPDX-License-Identifier: MIT
#include "primitives_n216_common.hpp"
#if N216_LIBRARY == 0
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/device/device_scan.hpp>
#include <rocprim/device/device_scan_by_key.hpp>
#include <rocprim/device/device_segmented_reduce.hpp>
#include <rocprim/device/device_segmented_scan.hpp>
constexpr const char *library = "rocPRIM";
#elif N216_LIBRARY == 1
#include <hipcub/device/device_reduce.hpp>
#include <hipcub/device/device_scan.hpp>
#include <hipcub/device/device_segmented_reduce.hpp>
constexpr const char *library = "hipCUB";
#else
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/system/hip/execution_policy.h>
constexpr const char *library = "rocThrust";
#endif

template <class T, class Op>
void reduce(const Buffer<T> &in, Buffer<T> &out, T init, Op op,
            hipStream_t stream) {
#if N216_LIBRARY == 0
  temporary(
      [&](void *p, size_t &bytes) {
        return rocprim::reduce(p, bytes, in.data, out.data, init, in.n, op,
                               stream);
      },
      stream);
#elif N216_LIBRARY == 1
  temporary(
      [&](void *p, size_t &bytes) {
        return hipcub::DeviceReduce::Reduce(p, bytes, in.data, out.data, in.n,
                                            op, init, stream);
      },
      stream);
#else
  auto first = thrust::device_pointer_cast(in.data);
  T result = thrust::reduce(thrust::hip::par_nosync.on(stream), first,
                            first + in.n, init, op);
  // Thrust's public reduction returns a host scalar; preserve its actual API.
  check(hipMemcpyAsync(out.data, &result, sizeof(T), hipMemcpyHostToDevice,
                       stream));
  check(hipStreamSynchronize(stream));
#endif
}
template <bool Exclusive, class T, class Op>
void scan(const Buffer<T> &in, Buffer<T> &out, T init, Op op,
          hipStream_t stream) {
#if N216_LIBRARY == 0
  temporary(
      [&](void *p, size_t &bytes) {
        if constexpr (Exclusive)
          return rocprim::exclusive_scan(p, bytes, in.data, out.data, init,
                                         in.n, op, stream);
        else
          return rocprim::inclusive_scan(p, bytes, in.data, out.data, in.n, op,
                                         stream);
      },
      stream);
#elif N216_LIBRARY == 1
  temporary(
      [&](void *p, size_t &bytes) {
        if constexpr (Exclusive)
          return hipcub::DeviceScan::ExclusiveScan(p, bytes, in.data, out.data,
                                                   op, init, in.n, stream);
        else
          return hipcub::DeviceScan::InclusiveScan(p, bytes, in.data, out.data,
                                                   op, in.n, stream);
      },
      stream);
#else
  auto first = thrust::device_pointer_cast(in.data),
       output = thrust::device_pointer_cast(out.data);
  if constexpr (Exclusive)
    thrust::exclusive_scan(thrust::hip::par_nosync.on(stream), first,
                           first + in.n, output, init, op);
  else
    thrust::inclusive_scan(thrust::hip::par_nosync.on(stream), first,
                           first + in.n, output, op);
#endif
}
template <bool Exclusive, class T, class Op>
void scan_by_key(const Buffer<int> &keys, const Buffer<T> &in, Buffer<T> &out,
                 T init, Op op, hipStream_t stream) {
#if N216_LIBRARY == 0
  temporary(
      [&](void *p, size_t &bytes) {
        if constexpr (Exclusive)
          return rocprim::exclusive_scan_by_key(p, bytes, keys.data, in.data,
                                                out.data, init, in.n, op,
                                                Equal(), stream);
        else
          return rocprim::inclusive_scan_by_key(p, bytes, keys.data, in.data,
                                                out.data, in.n, op, Equal(),
                                                stream);
      },
      stream);
#elif N216_LIBRARY == 1
  temporary(
      [&](void *p, size_t &bytes) {
        if constexpr (Exclusive)
          return hipcub::DeviceScan::ExclusiveScanByKey(
              p, bytes, keys.data, in.data, out.data, op, init, in.n, Equal(),
              stream);
        else
          return hipcub::DeviceScan::InclusiveScanByKey(p, bytes, keys.data,
                                                        in.data, out.data, op,
                                                        in.n, Equal(), stream);
      },
      stream);
#else
  auto first = thrust::device_pointer_cast(in.data),
       output = thrust::device_pointer_cast(out.data);
  auto key = thrust::device_pointer_cast(keys.data);
  if constexpr (Exclusive)
    thrust::exclusive_scan_by_key(thrust::hip::par_nosync.on(stream), key,
                                  key + in.n, first, output, init, Equal(), op);
  else
    thrust::inclusive_scan_by_key(thrust::hip::par_nosync.on(stream), key,
                                  key + in.n, first, output, Equal(), op);
#endif
}
// rocPRIM/hipCUB expose offsets, including empty segments. Thrust exposes
// reduce-by-key, so its result comprises nonempty contiguous key runs only.
template <class T, class Op>
size_t segmented_reduce(const Buffer<int> &bounds, const Buffer<int> &keys,
                        const Buffer<T> &in, Buffer<T> &out, T init, Op op,
                        hipStream_t stream) {
#if N216_LIBRARY == 0
  temporary(
      [&](void *p, size_t &bytes) {
        return rocprim::segmented_reduce(p, bytes, in.data, out.data,
                                         bounds.n - 1, bounds.data,
                                         bounds.data + 1, op, init, stream);
      },
      stream);
  return bounds.n - 1;
#elif N216_LIBRARY == 1
  temporary(
      [&](void *p, size_t &bytes) {
        return hipcub::DeviceSegmentedReduce::Reduce(
            p, bytes, in.data, out.data, bounds.n - 1, bounds.data,
            bounds.data + 1, op, init, stream);
      },
      stream);
  return bounds.n - 1;
#else
  // reduce_by_key has no init parameter; tests use its documented identity.
  (void)bounds;
  (void)init;
  Buffer<int> key_out(in.n);
  auto key = thrust::device_pointer_cast(keys.data),
       key_result = thrust::device_pointer_cast(key_out.data);
  auto result =
      thrust::reduce_by_key(thrust::hip::par_nosync.on(stream), key, key + in.n,
                            thrust::device_pointer_cast(in.data), key_result,
                            thrust::device_pointer_cast(out.data), Equal(), op);
  size_t count = result.first - key_result;
  auto actual = key_out.read(count, stream);
  std::vector<int> expected;
  auto source = keys.read(keys.n, stream);
  for (size_t i = 0; i < source.size(); ++i)
    if (i == 0 || source[i] != source[i - 1])
      expected.push_back(source[i]);
  require(actual == expected, "reduce-by-key keys/count mismatch");
  return count;
#endif
}
template <class T> void floating(size_t n, int pattern, hipStream_t stream) {
  std::vector<T> values(n);
  for (size_t i = 0; i < n; ++i)
    values[i] = T(pattern == 0 ? (int((i * 37 + 11) % 97) - 48) / 64.0L
                               : std::sin(0.13L * i) * 0.13L +
                                     (int(i % 11) - 5) * 0.02L);
  auto bounds = offsets(n), key_values = keys_from_offsets(bounds);
  Buffer<T> in(values, stream), out(n), scalar(1), sums(bounds.size() - 1);
  Buffer<int> bound_in(bounds, stream), keys(key_values, stream);
  // The input constructor with stream is explicit to avoid hidden defaults.
  Plus<T> op;
  const T init = T(0.75);
  std::vector<long double> inclusive(n), exclusive(n), by_inclusive(n),
      by_exclusive(n);
  long double total = 0, running = 0;
  for (size_t i = 0; i < n; ++i) {
    exclusive[i] = static_cast<long double>(init) + total;
    total += values[i];
    inclusive[i] = total;
    if (i == 0 || key_values[i] != key_values[i - 1])
      running = 0;
    by_exclusive[i] = static_cast<long double>(init) + running;
    running += values[i];
    by_inclusive[i] = running;
  }
  reduce(in, scalar, -init, op, stream);
  verify(scalar, {total - init}, stream, "reduce", pattern == 0);
  scan<false>(in, out, T(0), op, stream);
  verify(out, inclusive, stream, "inclusive", pattern == 0);
  scan<true>(in, out, init, op, stream);
  verify(out, exclusive, stream, "exclusive_init", pattern == 0);
  scan_by_key<false>(keys, in, out, T(0), op, stream);
  verify(out, by_inclusive, stream, "key_inclusive", pattern == 0);
  scan_by_key<true>(keys, in, out, init, op, stream);
  verify(out, by_exclusive, stream, "key_exclusive_init", pattern == 0);
  std::vector<long double> expected;
  for (size_t j = 0; j + 1 < bounds.size(); ++j) {
#if N216_LIBRARY == 2
    if (bounds[j] == bounds[j + 1])
      continue;
    long double sum = 0;
#else
    long double sum = -init;
#endif
    for (int k = bounds[j]; k < bounds[j + 1]; ++k)
      sum += values[k];
    expected.push_back(sum);
  }
  size_t count = segmented_reduce(bound_in, keys, in, sums, -init, op, stream);
  require(count == expected.size(), "segmented result count mismatch");
  verify(sums, expected, stream, "segmented_reduce", pattern == 0);
  Buffer<T> inplace(values, stream);
  scan<false>(inplace, inplace, T(0), op, stream);
  verify(inplace, inclusive, stream, "inclusive_inplace", pattern == 0);
  Buffer<T> inplace_key(values, stream);
  scan_by_key<true>(keys, inplace_key, inplace_key, init, op, stream);
  verify(inplace_key, by_exclusive, stream, "key_exclusive_inplace",
         pattern == 0);
#if N216_LIBRARY == 0
  temporary(
      [&](void *p, size_t &bytes) {
        return rocprim::segmented_inclusive_scan(
            p, bytes, in.data, out.data, bounds.size() - 1, bound_in.data,
            bound_in.data + 1, op, stream);
      },
      stream);
  verify(out, by_inclusive, stream, "offset_inclusive", pattern == 0);
  temporary(
      [&](void *p, size_t &bytes) {
        return rocprim::segmented_exclusive_scan(
            p, bytes, in.data, out.data, bounds.size() - 1, bound_in.data,
            bound_in.data + 1, init, op, stream);
      },
      stream);
  verify(out, by_exclusive, stream, "offset_exclusive_init", pattern == 0);
#endif
  in.unchanged(values, stream);
  keys.unchanged(key_values, stream);
  bound_in.unchanged(bounds, stream);
}
template <class T> void custom(size_t n, hipStream_t stream) {
  using A = ArgMax<T>;
  const A identity{-std::numeric_limits<T>::infinity(),
                   std::numeric_limits<int>::max()};
  std::vector<A> values(n), inclusive(n), by_key(n);
  auto bounds = offsets(n), key_values = keys_from_offsets(bounds);
  A best = identity, local = identity;
  // CPU oracle deliberately uses explicit value/index comparisons, not the
  // device reduction functor. Repeated maxima require first-index tie handling.
  for (size_t i = 0; i < n; ++i) {
    values[i] = {T(int((i * 7) % 13) - 6), int(i)};
    if (values[i].value > best.value)
      best = values[i];
    inclusive[i] = best;
    if (i == 0 || key_values[i] != key_values[i - 1])
      local = identity;
    if (values[i].value > local.value)
      local = values[i];
    by_key[i] = local;
  }
  Buffer<A> in(values, stream), out(n), scalar(1), sums(bounds.size() - 1);
  Buffer<int> bound_in(bounds, stream), keys(key_values, stream);
  ChooseMax<T> op;
  reduce(in, scalar, identity, op, stream);
  verify(scalar, std::vector<A>{best}, stream, "argmax_reduce");
  scan<false>(in, out, identity, op, stream);
  verify(out, inclusive, stream, "argmax_inclusive");
  scan_by_key<false>(keys, in, out, identity, op, stream);
  verify(out, by_key, stream, "argmax_key_inclusive");
  std::vector<A> expected;
  for (size_t j = 0; j + 1 < bounds.size(); ++j) {
#if N216_LIBRARY == 2
    if (bounds[j] == bounds[j + 1])
      continue;
#endif
    A candidate = identity;
    for (int k = bounds[j]; k < bounds[j + 1]; ++k)
      if (values[k].value > candidate.value)
        candidate = values[k];
    expected.push_back(candidate);
  }
  size_t count =
      segmented_reduce(bound_in, keys, in, sums, identity, op, stream);
  require(count == expected.size(), "argmax segmented count mismatch");
  verify(sums, expected, stream, "argmax_segmented_reduce");
  in.unchanged(values, stream);
  keys.unchanged(key_values, stream);
  bound_in.unchanged(bounds, stream);
}
int main() {
  try {
    native_device();
    Stream private_stream;
    for (int stream_index = 0; stream_index < 2; ++stream_index) {
      hipStream_t stream = stream_index ? private_stream.value : nullptr;
      for (size_t n : {size_t(0), size_t(1), size_t(33), size_t(257),
                       size_t(4099), size_t(65537)}) {
        for (int pattern = 0; pattern < 2; ++pattern) {
          context = "f32/n" + std::to_string(n) + "/pattern" +
                    std::to_string(pattern) + "/stream" +
                    std::to_string(stream_index);
          floating<float>(n, pattern, stream);
          context.replace(0, 3, "f64");
          floating<double>(n, pattern, stream);
        }
        context = "arg32/n" + std::to_string(n) + "/stream" +
                  std::to_string(stream_index);
        custom<float>(n, stream);
        context.replace(0, 5, "arg64");
        custom<double>(n, stream);
      }
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
