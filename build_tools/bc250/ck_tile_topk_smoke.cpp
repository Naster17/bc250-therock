// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "topk_softmax_api.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

template <typename T>
void run_case(const std::string &type, const std::string &activation,
              int experts, int k, bool saturated, bool exceptional = false) {
  constexpr int rows = 17, offset = 5;
  const int si = experts + 7, so = k + 5;
  std::vector<T> input(offset + rows * si + 7, ck_tile::type_convert<T>(29.f));
  std::vector<float> output(offset + rows * so + 7, -777.f);
  std::vector<ck_tile::index_t> indices(output.size(), -777);
  for (int r = 0; r < rows; ++r)
    for (int e = 0; e < experts; ++e)
      input[offset + r * si + e] = ck_tile::type_convert<T>(
          exceptional ? (e % 4 == 0   ? NAN
                         : e % 4 == 1 ? -INFINITY
                         : e % 4 == 2 ? INFINITY
                                      : 0.f)
          : saturated ? -1000.f
                      : float((e * 37 + r * 11) % 197 - 98) / 32.f);
  ck_tile::DeviceMem x(input.size() * sizeof(T));
  ck_tile::DeviceMem y(output.size() * sizeof(float));
  ck_tile::DeviceMem ix(indices.size() * sizeof(ck_tile::index_t));
  x.ToDevice(input.data());
  y.ToDevice(output.data());
  ix.ToDevice(indices.data());
  topk_softmax_kargs args;
  args.p_input = static_cast<T *>(x.GetDeviceBuffer()) + offset;
  args.p_output = static_cast<float *>(y.GetDeviceBuffer()) + offset;
  args.p_indices =
      static_cast<ck_tile::index_t *>(ix.GetDeviceBuffer()) + offset;
  args.num_rows = rows;
  args.num_experts = experts;
  args.topk = k;
  args.stride_input = si;
  args.stride_output = so;
  require(topk_softmax({type, "fp32", experts, activation}, args,
                       {nullptr, false, 0, 0, 1}) >= 0,
          "top-k dispatch rejected");
  HIP_CHECK_ERROR(hipDeviceSynchronize());
  y.FromDevice(output.data());
  ix.FromDevice(indices.data());
  for (int r = 0; r < rows; ++r) {
    std::vector<double> ref(experts);
    double maximum = -INFINITY, sum = 0;
    for (int e = 0; e < experts; ++e) {
      ref[e] = ck_tile::type_convert<float>(input[offset + r * si + e]);
      maximum = std::max(maximum, ref[e]);
    }
    if (activation == "softmax") {
      for (auto &v : ref) {
        v = std::exp(v - maximum);
        sum += v;
      }
      for (auto &v : ref)
        v /= sum;
    }
    if (activation == "sigmoid")
      for (auto &v : ref)
        v = 1.0 / (1.0 + std::exp(-v));
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const bool na = std::isnan(ref[a]), nb = std::isnan(ref[b]);
      if (na != nb)
        return na;
      if (na && nb)
        return a < b;
      return ref[a] > ref[b] || (ref[a] == ref[b] && a < b);
    });
    for (int i = 0; i < k; ++i) {
      const int p = offset + r * so + i;
      const auto expected = ref[order[i]];
      const bool value_ok = std::isnan(expected) ? std::isnan(output[p])
                            : std::isinf(expected)
                                ? output[p] == expected
                                : std::isfinite(output[p]) &&
                                      std::abs(output[p] - expected) <=
                                          2.e-5 + std::abs(expected) * 2.e-5;
      if (indices[p] != order[i] || !value_ok) {
        std::cerr << type << ' ' << activation << " experts=" << experts
                  << " row=" << r << " k=" << i << " index=" << indices[p]
                  << " expected=" << order[i] << " value=" << output[p] << '\n';
        throw std::runtime_error("top-k CPU reference mismatch");
      }
    }
  }
  for (std::size_t p = 0; p < output.size(); ++p) {
    const auto relative = static_cast<long>(p) - offset;
    const bool valid =
        relative >= 0 && relative < rows * so && relative % so < k;
    if (!valid)
      require(output[p] == -777.f && indices[p] == -777,
              "top-k wrote output padding/guards");
  }
}

template <typename T> int run_type(const std::string &type) {
  int cases = 0;
  for (const auto &activation : {"identity", "softmax", "sigmoid"})
    for (int experts : {7, 31, 33, 65, 129, 191}) {
      run_case<T>(type, activation, experts, experts == 7 ? 7 : 5, false);
      ++cases;
    }
  // Valid sigmoid zeroes must beat padded experts, even when every input ties.
  run_case<T>(type, "sigmoid", 7, 7, true);
  run_case<T>(type, "identity", 7, 7, false, true);
  return cases + 2;
}

int main() {
  try {
    hipDeviceProp_t prop{};
    HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, 0));
    require(std::string(prop.gcnArchName).starts_with("gfx1013") &&
                prop.warpSize == 32,
            "expected native gfx1013 wave32");
    const int cases = run_type<ck_tile::fp16_t>("fp16") +
                      run_type<ck_tile::bf16_t>("bf16") +
                      run_type<float>("fp32");
    std::cout << "CK_TILE_TOPK_PASS cases=" << cases << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
