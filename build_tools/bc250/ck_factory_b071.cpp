// SPDX-License-Identifier: MIT
// Audit the existing MIOpen-layout FP32 factory; no plugin or routing changes.
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#include "primitives_n216_common.hpp"
#include <array>
#include <iostream>
namespace dev = ck::tensor_operation::device;
namespace ew = ck::tensor_operation::element_wise;
namespace layout = ck::tensor_layout::convolution;
using Op = dev::DeviceGroupedConvFwdMultipleABD<
    2, layout::NHWGC, layout::GKYXC, ck::Tuple<>, layout::NHWGK, float, float,
    ck::Tuple<>, float, ew::PassThrough, ew::PassThrough, ew::PassThrough>;
using Factory = dev::instance::DeviceOperationInstanceFactory<Op>;
using A5 = std::array<ck::index_t, 5>;
using A2 = std::array<ck::index_t, 2>;
int main() {
  try {
    hipDeviceProp_t props{};
    check(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName) == "gfx1013:xnack-", "architecture");
    auto ops = Factory::GetInstances();
#ifndef DL_KERNELS
    require(ops.empty(), "unexpected factory without DL macro");
    std::cout << "{\"status\":\"passed\",\"instances\":0,\"executions\":0}\n";
#else
    require(!ops.empty(), "missing installed scalar DL instances");
    Stream private_stream;
    size_t executions = 0;
    double max_error = 0;
    for (hipStream_t stream : {hipStream_t{}, private_stream.value})
      for (int groups : {1, 3})
        for (int filter : {1, 3})
          for (bool dyadic : {false, true}) {
            const int n = 2, c = 4, k = 4, h = 5, w = 7, pad = filter / 2;
            A5 al{groups, n, c, h, w}, bl{groups, k, c, filter, filter};
            A5 el{groups, n, k, h, w};
            A5 as{c, h * w * groups * c, 1, w * groups * c, groups * c};
            A5 bs{k * filter * filter * c, filter * filter * c, 1, filter * c,
                  c};
            A5 es{k, h * w * groups * k, 1, w * groups * k, groups * k};
            auto at = [](A5 s, int g, int b, int ch, int y, int x) {
              return g * s[0] + b * s[1] + ch + y * s[3] + x * s[4];
            };
            std::vector<float> a(groups * n * c * h * w);
            std::vector<float> b(groups * k * c * filter * filter);
            for (size_t i = 0; i < a.size(); ++i)
              a[i] = float(int((i * 13 + 3) % 19) - 9) / (dyadic ? 8.f : 11.f);
            for (size_t i = 0; i < b.size(); ++i)
              b[i] = float(int((i * 7 + 5) % 17) - 8) / (dyadic ? 8.f : 11.f);
            std::vector<double> gold(groups * n * k * h * w);
            for (int g = 0; g < groups; ++g)
              for (int batch = 0; batch < n; ++batch)
                for (int out = 0; out < k; ++out)
                  for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                      double sum = 0;
                      for (int ch = 0; ch < c; ++ch)
                        for (int fy = 0; fy < filter; ++fy)
                          for (int fx = 0; fx < filter; ++fx) {
                            int iy = y + fy - pad, ix = x + fx - pad;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                              sum += double(a[at(as, g, batch, ch, iy, ix)]) *
                                     double(b[at(bs, g, out, ch, fy, fx)]);
                          }
                      gold[at(es, g, batch, out, y, x)] = sum;
                    }
            Buffer<float> da(a, stream), db(b, stream);
            size_t admitted = 0;
            for (auto &op : ops) {
              Buffer<float> de(gold.size());
              auto arg = op->MakeArgumentPointer(
                  da.data, db.data, std::array<const void *, 0>{}, de.data, al,
                  as, bl, bs, std::array<A5, 0>{}, std::array<A5, 0>{}, el, es,
                  A2{1, 1}, A2{1, 1}, A2{pad, pad}, A2{pad, pad},
                  ew::PassThrough{}, ew::PassThrough{}, ew::PassThrough{});
              if (!op->IsSupportedArgument(arg.get()))
                continue;
              op->MakeInvokerPointer()->Run(arg.get(), {stream, false});
              auto actual = de.read(gold.size(), stream);
              for (size_t i = 0; i < gold.size(); ++i) {
                double error = std::abs(double(actual[i]) - gold[i]);
                require(std::isfinite(actual[i]) &&
                            error <=
                                (dyadic ? 0. : 2e-5 + 5e-5 * std::abs(gold[i])),
                        "CPU reference mismatch: " + op->GetTypeString());
                max_error = std::max(max_error, error);
              }
              da.unchanged(a, stream);
              db.unchanged(b, stream);
              ++admitted;
              ++executions;
            }
            require(admitted > 0, "no admitted instance for fixture");
          }
    std::cout << "{\"status\":\"passed\",\"architecture\":\"gfx1013\","
                 "\"fixtures\":16,"
              << "\"instances\":" << ops.size()
              << ",\"executions\":" << executions
              << ",\"max_abs_error\":" << max_error << ",\"guards\":true}\n";
#endif
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
