// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "priority2_primitives_common.hpp"
#include <hipcub/hipcub.hpp>
__global__ void block_ops(const int *in, int *scan, int *sum, int *warps) {
  using Scan = hipcub::BlockScan<int, 128>;
  using Reduce = hipcub::BlockReduce<int, 128>;
  using Warp = hipcub::WarpReduce<int, 32>;
  __shared__ Scan::TempStorage scan_storage;
  __shared__ Reduce::TempStorage reduce_storage;
  __shared__ Warp::TempStorage warp_storage[4];
  int lane = threadIdx.x;
  int x = in[blockIdx.x * 128 + lane], y;
  Scan(scan_storage).InclusiveSum(x, y);
  scan[blockIdx.x * 128 + lane] = y;
  int total = Reduce(reduce_storage).Sum(x);
  if (lane == 0)
    sum[blockIdx.x] = total;
  int warp = Warp(warp_storage[lane / 32]).Sum(x);
  if (lane % 32 == 0)
    warps[blockIdx.x * 4 + lane / 32] = warp;
}
int main() {
  try {
    native_device();
    hipStream_t custom{};
    check(hipStreamCreateWithFlags(&custom, hipStreamNonBlocking));
    for (auto stream : {hipStream_t(nullptr), custom}) {
      for (size_t n : {size_t(1), size_t(129), size_t(4099)}) {
        auto v = input(n);
        Buffer in(n, stream), out(n, stream), count(1, stream);
        in.upload(v, stream);
        std::vector<int> expected(n);
        temporary([&](void *p, size_t &s) {
          return hipcub::DeviceScan::InclusiveSum(p, s, in.data, out.data, n,
                                                  stream);
        });
        std::partial_sum(v.begin(), v.end(), expected.begin());
        verify(out, expected, stream, "inclusive scan");
        temporary([&](void *p, size_t &s) {
          return hipcub::DeviceScan::ExclusiveSum(p, s, in.data, out.data, n,
                                                  stream);
        });
        int running = 0;
        for (size_t i = 0; i < n; ++i) {
          expected[i] = running;
          running += v[i];
        }
        verify(out, expected, stream, "exclusive scan");
        temporary([&](void *p, size_t &s) {
          return hipcub::DeviceRadixSort::SortKeys(p, s, in.data, out.data, n,
                                                   0, 32, stream);
        });
        expected = v;
        std::sort(expected.begin(), expected.end());
        verify(out, expected, stream, "signed radix sort");
        temporary([&](void *p, size_t &s) {
          return hipcub::DeviceSelect::If(p, s, in.data, out.data, count.data,
                                          n, Even(), stream);
        });
        expected.clear();
        std::copy_if(v.begin(), v.end(), std::back_inserter(expected), Even());
        verify(count, {int(expected.size())}, stream, "selected count");
        verify(out, expected, stream, "stable selection");
        temporary([&](void *p, size_t &s) {
          return hipcub::DeviceReduce::Sum(p, s, in.data, out.data, n, stream);
        });
        verify(out, {std::accumulate(v.begin(), v.end(), 0)}, stream, "reduce");
        in.read(stream);
      }
      for (int blocks : {1, 5}) {
        auto v = input(blocks * 128);
        Buffer in(v.size(), stream), scan(v.size(), stream),
            sums(blocks, stream), warps(blocks * 4, stream);
        in.upload(v, stream);
        hipLaunchKernelGGL(block_ops, dim3(blocks), dim3(128), 0, stream,
                           in.data, scan.data, sums.data, warps.data);
        check(hipGetLastError());
        std::vector<int> expected(v.size()), block_sum(blocks),
            warp_sum(blocks * 4);
        for (int b = 0; b < blocks; ++b) {
          std::partial_sum(v.begin() + b * 128, v.begin() + (b + 1) * 128,
                           expected.begin() + b * 128);
          block_sum[b] = expected[(b + 1) * 128 - 1];
          for (int w = 0; w < 4; ++w)
            warp_sum[b * 4 + w] =
                std::accumulate(v.begin() + b * 128 + w * 32,
                                v.begin() + b * 128 + (w + 1) * 32, 0);
        }
        verify(scan, expected, stream, "block scan");
        verify(sums, block_sum, stream, "block reduction");
        verify(warps, warp_sum, stream, "wave32 warp reduction");
        in.read(stream);
      }
    }
    check(hipStreamDestroy(custom));
    return finish("hipCUB");
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
