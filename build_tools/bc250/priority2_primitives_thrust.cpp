// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "priority2_primitives_common.hpp"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

template <class Policy> void run(Policy policy, hipStream_t stream) {
  for (size_t n : {size_t(1), size_t(129), size_t(4099)}) {
    auto v = input(n);
    Buffer in(n, stream), out(n, stream), vals(n, stream);
    in.upload(v, stream);
    auto begin = thrust::device_pointer_cast(in.data),
         output = thrust::device_pointer_cast(out.data),
         values = thrust::device_pointer_cast(vals.data);
    std::vector<int> expected(n), ids(n);
    thrust::inclusive_scan(policy, begin, begin + n, output);
    std::partial_sum(v.begin(), v.end(), expected.begin());
    verify(out, expected, stream, "inclusive scan");
    thrust::exclusive_scan(policy, begin, begin + n, output, 17);
    int running = 17;
    for (size_t i = 0; i < n; ++i) {
      expected[i] = running;
      running += v[i];
    }
    verify(out, expected, stream, "exclusive scan");
    thrust::transform(policy, begin, begin + n, output, Twice());
    std::transform(v.begin(), v.end(), expected.begin(), Twice());
    verify(out, expected, stream, "transform");
    auto end = thrust::copy_if(policy, begin, begin + n, output, Even());
    expected.clear();
    std::copy_if(v.begin(), v.end(), std::back_inserter(expected), Even());
    if (size_t(end - output) != expected.size())
      throw std::runtime_error("selection size mismatch");
    verify(out, expected, stream, "selection");
    if (thrust::reduce(policy, begin, begin + n, 0) !=
        std::accumulate(v.begin(), v.end(), 0))
      throw std::runtime_error("reduction mismatch");
    ++cases;
    std::iota(ids.begin(), ids.end(), 0);
    vals.upload(ids, stream);
    thrust::stable_sort_by_key(policy, begin, begin + n, values);
    std::stable_sort(ids.begin(), ids.end(),
                     [&](int a, int b) { return v[a] < v[b]; });
    expected.resize(n);
    for (size_t i = 0; i < n; ++i)
      expected[i] = v[ids[i]];
    verify(in, expected, stream, "stable signed keys");
    verify(vals, ids, stream, "stable values");
    end = thrust::unique(policy, begin, begin + n);
    expected.erase(std::unique(expected.begin(), expected.end()),
                   expected.end());
    if (size_t(end - begin) != expected.size())
      throw std::runtime_error("unique size mismatch");
    verify(in, expected, stream, "unique");
  }
}
int main() {
  try {
    native_device();
    run(thrust::hip::par.on(nullptr), nullptr);
    hipStream_t stream{};
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    run(thrust::hip::par_nosync.on(stream), stream);
    check(hipStreamDestroy(stream));
    return finish("rocThrust");
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
