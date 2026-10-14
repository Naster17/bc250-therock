// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

template <typename T, bool Fused> void run_case(int n) {
  constexpr int m = 7, offset = 5;
  const int stride = n + 7;
  const T guard = ck_tile::type_convert<T>(-7.f);
  std::vector<T> x(offset + m * stride + 7, guard), residual(x.size(), guard);
  std::vector<T> y(x.size(), guard), gamma(n), beta(n), bias(n);
  std::vector<float> mean(m + 2, -777.f), inv(m + 2, -777.f);
  for (int c = 0; c < n; ++c) {
    gamma[c] = ck_tile::type_convert<T>(float(c % 9 - 4) / 16.f + 1.f);
    beta[c] = ck_tile::type_convert<T>(float(c % 7 - 3) / 16.f);
    bias[c] = ck_tile::type_convert<T>(float(c % 5 - 2) / 32.f);
    for (int r = 0; r < m; ++r) {
      x[offset + r * stride + c] =
          ck_tile::type_convert<T>(float((r * 17 + c * 13) % 89 - 44) / 16.f);
      residual[offset + r * stride + c] =
          ck_tile::type_convert<T>(float((r + c * 3) % 17 - 8) / 32.f);
    }
  }
  ck_tile::DeviceMem dx(x.size() * sizeof(T)), dr(x.size() * sizeof(T)),
      dy(y.size() * sizeof(T));
  ck_tile::DeviceMem dg(n * sizeof(T)), db(n * sizeof(T)), dxb(n * sizeof(T));
  ck_tile::DeviceMem dm(mean.size() * sizeof(float)),
      di(inv.size() * sizeof(float));
  dx.ToDevice(x.data());
  dr.ToDevice(residual.data());
  dy.ToDevice(y.data());
  dg.ToDevice(gamma.data());
  db.ToDevice(beta.data());
  dxb.ToDevice(bias.data());
  dm.ToDevice(mean.data());
  di.ToDevice(inv.data());
  using Shape = ck_tile::Generic2dBlockShape<ck_tile::sequence<2, 128>,
                                             ck_tile::sequence<2, 128>,
                                             ck_tile::sequence<1, 1>>;
  using Traits = ck_tile::Layernorm2dFwdTraits<
      true, true, false, true, true,
      Fused ? ck_tile::Layernorm2dXBiasEnum::ADD_BIAS
            : ck_tile::Layernorm2dXBiasEnum::NO_BIAS,
      Fused ? ck_tile::Layernorm2dFusedAddEnum::PRE_ADD
            : ck_tile::Layernorm2dFusedAddEnum::NO_ADD,
      ck_tile::Layernorm2dFusedQuantEnum::NO_SWEEP>;
  using Problem =
      ck_tile::Layernorm2dFwdPipelineProblem<T, T, T, T, float, T, float, float,
                                             ck_tile::null_type,
                                             ck_tile::null_type, Shape, Traits>;
  using Epilogue = ck_tile::Default2DEpilogue<
      ck_tile::Default2DEpilogueProblem<float, T, false, true, true>>;
  using Kernel =
      ck_tile::Layernorm2dFwd<ck_tile::Layernorm2dFwdPipelineTwoPass<Problem>,
                              Epilogue>;
  ck_tile::Layernorm2dFwdHostArgs args{
      static_cast<T *>(dx.GetDeviceBuffer()) + offset,
      static_cast<T *>(dr.GetDeviceBuffer()) + offset,
      nullptr,
      dxb.GetDeviceBuffer(),
      dg.GetDeviceBuffer(),
      db.GetDeviceBuffer(),
      static_cast<T *>(dy.GetDeviceBuffer()) + offset,
      nullptr,
      nullptr,
      static_cast<float *>(dm.GetDeviceBuffer()) + 1,
      static_cast<float *>(di.GetDeviceBuffer()) + 1,
      1.e-5f,
      m,
      n,
      stride,
      stride,
      stride,
      stride};
  ck_tile::launch_kernel(
      {nullptr, false, 0, 0, 1},
      ck_tile::make_kernel<1>(Kernel{}, Kernel::GridSize(args),
                              Kernel::BlockSize(), 0, Kernel::MakeKargs(args)));
  HIP_CHECK_ERROR(hipDeviceSynchronize());
  dy.FromDevice(y.data());
  dm.FromDevice(mean.data());
  di.FromDevice(inv.data());
  for (int r = 0; r < m; ++r) {
    std::vector<double> values(n);
    double mu = 0, variance = 0;
    for (int c = 0; c < n; ++c) {
      const auto p = offset + r * stride + c;
      values[c] = ck_tile::type_convert<float>(x[p]);
      if constexpr (Fused)
        values[c] += ck_tile::type_convert<float>(bias[c]) +
                     ck_tile::type_convert<float>(residual[p]);
      mu += values[c];
    }
    mu /= n;
    for (auto v : values)
      variance += (v - mu) * (v - mu);
    const double istd = 1.0 / std::sqrt(variance / n + 1.e-5);
    require(std::abs(mean[r + 1] - mu) < 2.e-5 &&
                std::abs(inv[r + 1] - istd) < 2.e-5,
            "LayerNorm mean/invstd CPU mismatch");
    for (int c = 0; c < n; ++c) {
      const double value =
          (values[c] - mu) * istd * ck_tile::type_convert<float>(gamma[c]) +
          ck_tile::type_convert<float>(beta[c]);
      const float expected =
          ck_tile::type_convert<float>(ck_tile::type_convert<T>(float(value)));
      const float actual =
          ck_tile::type_convert<float>(y[offset + r * stride + c]);
      const double tolerance = std::is_same_v<T, float> ? 2.e-5 : 0.016;
      require(std::abs(actual - expected) < tolerance,
              "LayerNorm output CPU mismatch");
    }
  }
  for (std::size_t p = 0; p < y.size(); ++p) {
    const auto relative = static_cast<long>(p) - offset;
    const bool valid =
        relative >= 0 && relative < m * stride && relative % stride < n;
    if (!valid)
      require(ck_tile::type_convert<float>(y[p]) == -7.f,
              "LayerNorm wrote padding/guards");
  }
  require(mean.front() == -777.f && mean.back() == -777.f &&
              inv.front() == -777.f && inv.back() == -777.f,
          "LayerNorm wrote statistics guards");
}

template <typename T> int run_type() {
  for (int n : {17, 129, 513}) {
    run_case<T, false>(n);
    run_case<T, true>(n);
  }
  return 6;
}

int main() {
  try {
    hipDeviceProp_t prop{};
    HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, 0));
    require(std::string(prop.gcnArchName).starts_with("gfx1013") &&
                prop.warpSize == 32,
            "expected native gfx1013 wave32");
    const int cases = run_type<ck_tile::fp16_t>() +
                      run_type<ck_tile::bf16_t>() + run_type<float>();
    std::cout << "CK_TILE_LAYERNORM_PASS cases=" << cases << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
