// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "priority2_primitives_common.hpp"
#include <rocprim/rocprim.hpp>
int main() {
  try {
    native_device();
    hipStream_t custom{};
    check(hipStreamCreateWithFlags(&custom, hipStreamNonBlocking));
    for (auto stream : {hipStream_t(nullptr), custom}) {
      for (size_t n : {size_t(1), size_t(129), size_t(4099)}) {
        auto v = input(n);
        Buffer in(n, stream), out(n, stream), vals(n, stream),
            sorted_vals(n, stream), count(1, stream);
        in.upload(v, stream);
        std::vector<int> expected(n), ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        vals.upload(ids, stream);
        temporary([&](void *p, size_t &s) {
          return rocprim::inclusive_scan(p, s, in.data, out.data, n,
                                         rocprim::plus<int>(), stream);
        });
        std::partial_sum(v.begin(), v.end(), expected.begin());
        verify(out, expected, stream, "inclusive scan");
        temporary([&](void *p, size_t &s) {
          return rocprim::exclusive_scan(p, s, in.data, out.data, 17, n,
                                         rocprim::plus<int>(), stream);
        });
        int running = 17;
        for (size_t i = 0; i < n; ++i) {
          expected[i] = running;
          running += v[i];
        }
        verify(out, expected, stream, "exclusive scan");
        temporary([&](void *p, size_t &s) {
          return rocprim::radix_sort_pairs(p, s, in.data, out.data, vals.data,
                                           sorted_vals.data, n, 0, 32, stream);
        });
        std::stable_sort(ids.begin(), ids.end(),
                         [&](int a, int b) { return v[a] < v[b]; });
        for (size_t i = 0; i < n; ++i)
          expected[i] = v[ids[i]];
        verify(out, expected, stream, "stable signed radix keys");
        verify(sorted_vals, ids, stream, "stable radix values");
        temporary([&](void *p, size_t &s) {
          return rocprim::select(p, s, in.data, out.data, count.data, n, Even(),
                                 stream);
        });
        expected.clear();
        std::copy_if(v.begin(), v.end(), std::back_inserter(expected), Even());
        verify(count, {int(expected.size())}, stream, "selected count");
        verify(out, expected, stream, "stable selection");
        temporary([&](void *p, size_t &s) {
          return rocprim::reduce(p, s, in.data, out.data, 0, n,
                                 rocprim::plus<int>(), stream);
        });
        verify(out, {std::accumulate(v.begin(), v.end(), 0)}, stream, "reduce");
        in.read(stream);
        vals.read(stream);
      }
    }
    check(hipStreamDestroy(custom));
    return finish("rocPRIM");
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
