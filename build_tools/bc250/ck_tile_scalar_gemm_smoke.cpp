// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/kernel/scalar_gemm_kernel.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

template <typename T, typename O>
void run_case(int layout, int k, bool broadcast, float alpha, float beta) {
  using Kernel = ck_tile::ScalarGemmKernel<T, O>;
  constexpr int m = 19, n = 27, batches = 3, offset = 5;
  const int lda = ((layout & 1) ? m : k) + 7;
  const int ldb = ((layout & 2) ? k : n) + 7;
  const int ldc = ((layout & 4) ? m : n) + 7;
  const int ar = (layout & 1) ? 1 : lda, ac = (layout & 1) ? lda : 1;
  const int br = (layout & 2) ? 1 : ldb, bc = (layout & 2) ? ldb : 1;
  const int cr = (layout & 4) ? 1 : ldc, cc = (layout & 4) ? ldc : 1;
  const int as = m * ar + k * ac + 9, bs = k * br + n * bc + 9;
  const int cs = m * cr + n * cc + 9;
  const T input_guard = ck_tile::type_convert<T>(17.f);
  const O output_guard = ck_tile::type_convert<O>(-7.f);
  std::vector<T> a(offset + batches * as + 7, input_guard);
  std::vector<T> b(offset + batches * bs + 7, input_guard);
  std::vector<O> c(offset + batches * cs + 7, output_guard);
  std::vector<bool> valid(c.size(), false);
  for (int batch = 0; batch < batches; ++batch) {
    for (int row = 0; row < m; ++row)
      for (int kk = 0; kk < k; ++kk)
        a[offset + batch * as + row * ar + kk * ac] = ck_tile::type_convert<T>(
            float((batch * 3 + row * 7 + kk * 11) % 31 - 15) / 16.f);
    for (int kk = 0; kk < k; ++kk)
      for (int col = 0; col < n; ++col)
        b[offset + batch * bs + kk * br + col * bc] = ck_tile::type_convert<T>(
            float((batch * 5 + col * 13 + kk * 7) % 37 - 18) / 16.f);
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col) {
        const int p = offset + batch * cs + row * cr + col * cc;
        valid[p] = true;
        c[p] = ck_tile::type_convert<O>(
            beta == 0.f ? NAN : float((row + col) % 7 - 3) / 8.f);
      }
  }
  const auto old = c;
  ck_tile::DeviceMem da(a.size() * sizeof(T)), db(b.size() * sizeof(T)),
      dc(c.size() * sizeof(O));
  da.ToDevice(a.data());
  db.ToDevice(b.data());
  dc.ToDevice(c.data());
  typename Kernel::Hargs args{
      alpha == 0.f ? nullptr : static_cast<T *>(da.GetDeviceBuffer()) + offset,
      alpha == 0.f ? nullptr : static_cast<T *>(db.GetDeviceBuffer()) + offset,
      static_cast<O *>(dc.GetDeviceBuffer()) + offset,
      m,
      n,
      k,
      batches,
      ar,
      ac,
      br,
      bc,
      cr,
      cc,
      broadcast ? 0 : as,
      broadcast ? 0 : bs,
      cs,
      alpha,
      beta};
  require(Kernel::IsSupportedArgument(args), "scalar GEMM rejected valid case");
  ck_tile::launch_kernel(
      {nullptr, false, 0, 0, 1},
      ck_tile::make_kernel<1>(Kernel{}, Kernel::GridSize(args),
                              Kernel::BlockSize(), 0, Kernel::MakeKargs(args)));
  HIP_CHECK_ERROR(hipDeviceSynchronize());
  dc.FromDevice(c.data());
  for (int batch = 0; batch < batches; ++batch)
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col) {
        double dot = 0;
        if (alpha != 0.f)
          for (int kk = 0; kk < k; ++kk)
            dot += double(ck_tile::type_convert<float>(
                       a[offset + (broadcast ? 0 : batch * as) + row * ar +
                         kk * ac])) *
                   ck_tile::type_convert<float>(
                       b[offset + (broadcast ? 0 : batch * bs) + kk * br +
                         col * bc]);
        const int p = offset + batch * cs + row * cr + col * cc;
        double result = alpha * dot;
        if (beta != 0.f)
          result += beta * ck_tile::type_convert<float>(old[p]);
        const float expected = ck_tile::type_convert<float>(
            ck_tile::type_convert<O>(float(result)));
        const float actual = ck_tile::type_convert<float>(c[p]);
        const double tolerance = std::is_same_v<O, float> ? 2.e-5 : 0.032;
        require(std::isfinite(actual) &&
                    std::abs(actual - expected) < tolerance,
                "scalar GEMM CPU-double reference mismatch");
      }
  for (std::size_t p = 0; p < c.size(); ++p)
    if (!valid[p])
      require(ck_tile::type_convert<float>(c[p]) == -7.f,
              "scalar GEMM wrote padding/guards");
  auto invalid = args;
  invalid.c_batch = -1;
  require(!Kernel::IsSupportedArgument(invalid),
          "scalar GEMM accepted negative batch stride");
  if (alpha != 0.f && k > 0) {
    invalid = args;
    invalid.a = reinterpret_cast<const T *>(args.c);
    require(!Kernel::IsSupportedArgument(invalid),
            "scalar GEMM accepted A/C overlap");
  }
  args.c_row = args.c_col = 1;
  require(!Kernel::IsSupportedArgument(args),
          "scalar GEMM accepted overlapping C");
}

template <typename T, typename O> int run_type() {
  for (int layout = 0; layout < 8; ++layout) {
    run_case<T, O>(layout, 17, false, 0.75f, -0.5f);
    run_case<T, O>(layout, 33, true, 1.f, 0.f);
    run_case<T, O>(layout, 0, false, 1.f, 0.5f);
    run_case<T, O>(layout, 17, false, 0.f, 0.f);
  }
  return 32;
}

int main() {
  try {
    hipDeviceProp_t prop{};
    HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, 0));
    require(std::string(prop.gcnArchName).starts_with("gfx1013") &&
                prop.warpSize == 32,
            "expected native gfx1013 wave32");
    const int cases = run_type<float, float>() +
                      run_type<ck_tile::fp16_t, ck_tile::fp16_t>() +
                      run_type<ck_tile::bf16_t, ck_tile::bf16_t>() +
                      run_type<ck_tile::fp16_t, float>() +
                      run_type<ck_tile::bf16_t, float>();
    std::cout << "CK_TILE_SCALAR_GEMM_PASS cases=" << cases << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
