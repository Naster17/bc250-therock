// SPDX-License-Identifier: MIT
// Bounded direct CK scalar DL qualification; no Torch/MIOpen dispatch change.
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "primitives_n216_common.hpp"
#include <array>
#include <iostream>

namespace dev = ck::tensor_operation::device;
namespace ew = ck::tensor_operation::element_wise;
namespace layout = ck::tensor_layout::convolution;
template <ck::index_t... I> using S = ck::Sequence<I...>;
// Same scalar tiling as upstream example/09_convnd_fwd/convnd_fwd_dl_fp32.cpp.
template <bool Fused>
using Conv = dev::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
    2, float, float, std::conditional_t<Fused, ck::Tuple<float>, ck::Tuple<>>,
    float, float, layout::G_NHW_C, layout::G_K_YX_C,
    std::conditional_t<Fused, ck::Tuple<layout::G_NHW_K>, ck::Tuple<>>,
    layout::G_NHW_K, ew::PassThrough, ew::PassThrough,
    std::conditional_t<Fused, ew::AddRelu, ew::PassThrough>,
    dev::ConvolutionForwardSpecialization::Default,
    dev::GemmSpecialization::MNKPadding, 256, 128, 128, 16, 1, 4, 4, 1, S<8, 2>,
    S<8, 2>, S<8, 1, 1, 1>, S<2, 1, 128, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>,
    S<4, 1, 1, 1>, S<1, 2, 0, 3>, S<1, 1, 1, 1>, S<8, 1, 1, 1>, S<2, 1, 128, 1>,
    S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 1>, S<1, 2, 0, 3>, S<1, 1, 1, 1>,
    S<0, 1, 2, 3, 4, 5>, 5, 4>;
using A5 = std::array<ck::index_t, 5>;
using A2 = std::array<ck::index_t, 2>;
struct Tensor {
  A5 length, stride;
  size_t size;
  Tensor(A5 l, bool padded) : length(l) {
    stride[2] = 1;
    // These tags flatten NHW (or YX) with one spatial stride. Keep that
    // dimension linear; arbitrary row/batch gaps are not this layout contract.
    stride[4] = l[2] + (padded ? 4 : 0);
    stride[3] = l[4] * stride[4];
    stride[1] = l[3] * stride[3];
    stride[0] = l[1] * stride[1] + (padded ? 16 : 0);
    size = 1;
    for (int i = 0; i < 5; ++i)
      size += (l[i] - 1) * stride[i];
  }
  size_t at(int g, int n, int c, int h, int w) const {
    return g * stride[0] + n * stride[1] + c + h * stride[3] + w * stride[4];
  }
};
struct Shape {
  int n, h, w, c, k, f, s, d, left, right;
};
static const std::array<Shape, 4> shapes{{{1, 5, 7, 4, 4, 1, 1, 1, 0, 0},
                                          {2, 7, 9, 8, 12, 3, 1, 1, 1, 1},
                                          {1, 9, 10, 4, 132, 3, 2, 1, 1, 0},
                                          {1, 8, 9, 8, 8, 3, 1, 2, 2, 1}}};
float value(size_t i, int salt, bool dyadic) {
  return float(int((i * 13 + salt * 7) % 19) - 9) / (dyadic ? 8.f : 11.f);
}
template <bool Fused>
void fixture(Shape q, int groups, bool padded, bool dyadic, hipStream_t stream,
             bool reject) {
  int oh = (q.h + q.left + q.right - q.d * (q.f - 1) - 1) / q.s + 1;
  int ow = (q.w + q.left + q.right - q.d * (q.f - 1) - 1) / q.s + 1;
  Tensor a({groups, q.n, q.c, q.h, q.w}, padded);
  Tensor b({groups, q.k, q.c, q.f, q.f}, padded);
  Tensor e({groups, q.n, q.k, oh, ow}, padded);
  std::vector<float> av(a.size), bv(b.size), dv(e.size);
  for (size_t i = 0; i < av.size(); ++i)
    av[i] = value(i, 1, dyadic);
  for (size_t i = 0; i < bv.size(); ++i)
    bv[i] = value(i, 2, dyadic);
  for (size_t i = 0; i < dv.size(); ++i)
    dv[i] = value(i, 3, dyadic);
  Buffer<float> da(av, stream), db(bv, stream), dd(dv, stream), de(e.size);
  std::array<const void *, Fused ? 1 : 0> ds{};
  std::array<A5, Fused ? 1 : 0> dl{}, dt{};
  if constexpr (Fused) {
    ds[0] = dd.data;
    dl[0] = e.length;
    dt[0] = e.stride;
  }
  auto arg = Conv<Fused>::MakeArgument(
      da.data, db.data, ds, de.data, a.length, a.stride, b.length, b.stride, dl,
      dt, e.length, e.stride, A2{q.s, q.s}, A2{q.d, q.d}, A2{q.left, q.left},
      A2{q.right, q.right}, ew::PassThrough{}, ew::PassThrough{},
      typename std::conditional_t<Fused, ew::AddRelu, ew::PassThrough>{});
  bool admitted = Conv<Fused>::IsSupportedArgument(arg);
  require(admitted != reject, "unexpected argument admission");
  if (reject) {
    de.read(0, stream);
    da.unchanged(av, stream);
    db.unchanged(bv, stream);
    dd.unchanged(dv, stream);
    passed("unsupported-vector-shape", 0);
    return;
  }
  Conv<Fused>::MakeInvoker().Run(arg, StreamConfig{stream, false});
  check(hipGetLastError());
  check(hipStreamSynchronize(stream));
  auto bytes = de.bytes(stream);
  std::vector<bool> written(e.size, false);
  long double error = 0;
  for (int g = 0; g < groups; ++g)
    for (int n = 0; n < q.n; ++n)
      for (int k = 0; k < q.k; ++k)
        for (int h = 0; h < oh; ++h)
          for (int w = 0; w < ow; ++w) {
            double sum = 0;
            for (int y = 0; y < q.f; ++y)
              for (int x = 0; x < q.f; ++x)
                for (int c = 0; c < q.c; ++c) {
                  int ih = h * q.s + y * q.d - q.left;
                  int iw = w * q.s + x * q.d - q.left;
                  if (ih >= 0 && ih < q.h && iw >= 0 && iw < q.w)
                    sum += double(av[a.at(g, n, c, ih, iw)]) *
                           bv[b.at(g, k, c, y, x)];
                }
            size_t pos = e.at(g, n, k, h, w);
            if constexpr (Fused)
              sum = std::max(0., sum + dv[pos]);
            float actual;
            std::memcpy(&actual,
                        bytes.data() + Buffer<float>::guard +
                            pos * sizeof(float),
                        sizeof(float));
            double diff = std::abs(double(actual) - sum);
            require(std::isfinite(actual) &&
                        diff <= (dyadic ? 0. : 2e-5 + 5e-5 * std::abs(sum)),
                    "CPU convolution reference mismatch: " + context +
                        "/offset" + std::to_string(pos) + "/actual" +
                        std::to_string(actual) + "/reference" +
                        std::to_string(sum));
            error = std::max(error, (long double)diff);
            written[pos] = true;
          }
  for (size_t i = 0; i < e.size; ++i)
    if (!written[i])
      for (size_t j = 0; j < sizeof(float); ++j)
        require(bytes[Buffer<float>::guard + i * sizeof(float) + j] == 0xa5,
                "output stride hole overwritten");
  da.unchanged(av, stream);
  db.unchanged(bv, stream);
  dd.unchanged(dv, stream);
  passed("cpu-reference", groups * q.n * q.k * oh * ow, error);
}
template <bool Fused> void run(hipStream_t stream, int sid) {
  for (int shape = 0; shape < 4; ++shape)
    for (int groups : {1, 3})
      for (bool padded : {false, true})
        for (bool dyadic : {false, true}) {
          context = std::string(Fused ? "add-relu" : "plain") + "/stream" +
                    std::to_string(sid) + "/shape" + std::to_string(shape) +
                    "/groups" + std::to_string(groups) + "/padded" +
                    std::to_string(padded) + "/dyadic" + std::to_string(dyadic);
          fixture<Fused>(shapes[shape], groups, padded, dyadic, stream, false);
        }
  for (int bad : {0, 1}) {
    Shape q = shapes[0];
    if (bad == 0)
      q.c = 3;
    else
      q.k = 5;
    context = std::string(Fused ? "add-relu" : "plain") + "/stream" +
              std::to_string(sid) + "/reject" + std::to_string(bad);
    fixture<Fused>(q, 1, false, true, stream, true);
  }
}
int main() {
  try {
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0));
    require(std::string(prop.gcnArchName).find("gfx1013") == 0,
            "native gfx1013 required");
    require(ck::is_gfx101_supported(),
            "scalar DL architecture admission missing");
    require(!ck::is_xdl_supported(), "XDL must remain unavailable");
    Stream private_stream;
    run<false>(nullptr, 0);
    run<true>(nullptr, 0);
    run<false>(private_stream.value, 1);
    run<true>(private_stream.value, 1);
    std::cout << "{\"status\":\"passed\",\"architecture\":\"gfx1013\","
                 "\"guards\":true,\"streams\":2,\"cases\":"
              << rows.size() << ",\"checks\":[";
    for (size_t i = 0; i < rows.size(); ++i)
      std::cout << (i ? "," : "") << rows[i];
    std::cout << "]}\n";
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
