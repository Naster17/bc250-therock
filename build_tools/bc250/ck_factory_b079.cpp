// SPDX-License-Identifier: MIT
// Isolated FP16 scalar-DL factories; installed FP32 plugin remains unchanged.
#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#include "primitives_n216_common.hpp"
#include <array>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
namespace dev = ck::tensor_operation::device;
namespace ew = ck::tensor_operation::element_wise;
namespace layout = ck::tensor_layout::convolution;
using H = ck::half_t;
using A5 = std::array<ck::index_t, 5>;
using A2 = std::array<ck::index_t, 2>;
template <bool Interleaved>
using Op = dev::DeviceGroupedConvFwdMultipleABD<
    2, std::conditional_t<Interleaved, layout::NHWGC, layout::GNHWC>,
    layout::GKYXC, ck::Tuple<>,
    std::conditional_t<Interleaved, layout::NHWGK, layout::GNHWK>, H, H,
    ck::Tuple<>, H, ew::PassThrough, ew::PassThrough, ew::PassThrough>;
H to_half(float value) { return ck::type_convert<H>(value); }
float full(H value) { return ck::type_convert<float>(value); }
__global__ void features(unsigned *out) {
  out[0] = 0;
  out[1] = 0;
#ifdef CK_USE_AMD_V_DOT2_F32_F16
  out[0] = 1;
#endif
#ifdef CK_USE_AMD_V_MAC_F32
  out[1] = 1;
#endif
  out[2] = warpSize;
}
struct Shape {
  int h, w, c, k, f, stride, dilation, pad;
  bool accumulator;
};
const std::array<Shape, 6> shapes{{{5, 7, 4, 4, 1, 1, 1, 0, false},
                                   {9, 11, 4, 8, 1, 2, 1, 0, false},
                                   {9, 10, 4, 132, 3, 2, 1, 1, false},
                                   {8, 9, 8, 8, 3, 1, 2, 2, false},
                                   {7, 9, 3, 5, 3, 1, 1, 1, false},
                                   {1, 1, 512, 4, 1, 1, 1, 0, true}}};
nlohmann::json result = {{"status", "running"},
                         {"layouts", nlohmann::json::object()}};
size_t executions = 0, refusals = 0, fixtures = 0, accumulator_executions = 0;
double max_error = 0, max_rounded_error = 0;
std::map<std::string, size_t> counts;
std::set<std::string> ids;
template <bool Interleaved> void run() {
  using Factory =
      dev::instance::DeviceOperationInstanceFactory<Op<Interleaved>>;
  auto ops = Factory::GetInstances();
  const std::string name = Interleaved ? "NHWGC" : "GNHWC";
#ifndef DL_KERNELS
  require(ops.empty(), "factory must be empty without DL_KERNELS");
  result["layouts"][name] = {{"instances", 0}};
#else
  require(ops.size() == 9, "expected nine installed FP16 scalar DL instances");
  std::set<std::string> unique;
  for (const auto &op : ops) {
    unique.insert(op->GetTypeString());
    ids.insert(op->GetTypeString());
  }
  require(unique.size() == 9, "FP16 IDs must be unambiguous including K1");
  result["layouts"][name] = {{"instances", ops.size()},
                             {"unique_ids", unique.size()}};
  Stream private_stream;
  for (hipStream_t stream : {hipStream_t{}, private_stream.value})
    for (int groups : {1, 3})
      for (const auto &q : shapes)
        for (int mode : {0, 1, 2}) {
          if ((mode == 2) != q.accumulator)
            continue;
          ++fixtures;
          const int n = 2;
          int oh =
              (q.h + 2 * q.pad - q.dilation * (q.f - 1) - 1) / q.stride + 1;
          int ow =
              (q.w + 2 * q.pad - q.dilation * (q.f - 1) - 1) / q.stride + 1;
          A5 al{groups, n, q.c, q.h, q.w}, bl{groups, q.k, q.c, q.f, q.f},
              el{groups, n, q.k, oh, ow};
          A5 as = Interleaved ? A5{q.c, q.h * q.w * groups * q.c, 1,
                                   q.w * groups * q.c, groups * q.c}
                              : A5{n * q.h * q.w * q.c, q.h * q.w * q.c, 1,
                                   q.w * q.c, q.c};
          A5 bs{q.k * q.f * q.f * q.c, q.f * q.f * q.c, 1, q.f * q.c, q.c};
          A5 es = Interleaved
                      ? A5{q.k, oh * ow * groups * q.k, 1, ow * groups * q.k,
                           groups * q.k}
                      : A5{n * oh * ow * q.k, oh * ow * q.k, 1, ow * q.k, q.k};
          auto at = [](A5 s, int g, int b, int c, int y, int x) {
            return g * s[0] + b * s[1] + c + y * s[3] + x * s[4];
          };
          std::vector<H> a(groups * n * q.c * q.h * q.w),
              b(groups * q.k * q.c * q.f * q.f);
          for (size_t i = 0; i < a.size(); ++i)
            a[i] = to_half(mode == 2 ? 1.f
                                     : float(int((i * 13 + 3) % 19) - 9) /
                                           (mode == 0 ? 8.f : 11.f));
          for (int g = 0; g < groups; ++g)
            for (int out = 0; out < q.k; ++out)
              for (int c = 0; c < q.c; ++c)
                for (int y = 0; y < q.f; ++y)
                  for (int x = 0; x < q.f; ++x) {
                    size_t i = at(bs, g, out, c, y, x);
                    b[i] =
                        to_half(mode == 2 ? (c < 256 ? 1.f : 1.f / 1024)
                                          : float(int((i * 7 + 5) % 17) - 8) /
                                                (mode == 0 ? 8.f : 11.f));
                  }
          std::vector<double> gold(groups * n * q.k * oh * ow);
          for (int g = 0; g < groups; ++g)
            for (int batch = 0; batch < n; ++batch)
              for (int out = 0; out < q.k; ++out)
                for (int y = 0; y < oh; ++y)
                  for (int x = 0; x < ow; ++x) {
                    double sum = 0;
                    for (int c = 0; c < q.c; ++c)
                      for (int fy = 0; fy < q.f; ++fy)
                        for (int fx = 0; fx < q.f; ++fx) {
                          int iy = y * q.stride + fy * q.dilation - q.pad,
                              ix = x * q.stride + fx * q.dilation - q.pad;
                          if (iy >= 0 && iy < q.h && ix >= 0 && ix < q.w)
                            sum +=
                                double(full(a[at(as, g, batch, c, iy, ix)])) *
                                double(full(b[at(bs, g, out, c, fy, fx)]));
                        }
                    gold[at(es, g, batch, out, y, x)] = sum;
                  }
          if (mode == 2) {
            H sequential = to_half(0);
            for (int c = 0; c < 512; ++c)
              sequential =
                  to_half(full(sequential) + (c < 256 ? 1.f : 1.f / 1024));
            require(gold[0] == 256.25 && full(sequential) == 256,
                    "accumulation discriminator fixture");
            result["accumulator_fixture"] = {{"FP64_sum", 256.25},
                                             {"sequential_FP16_sum", 256.0},
                                             {"expected_FP16_output", 256.25}};
          }
          Buffer<H> da(a, stream), db(b, stream);
          size_t admitted = 0;
          for (size_t ordinal = 0; ordinal < ops.size(); ++ordinal) {
            auto &op = ops[ordinal];
            Buffer<H> de(gold.size());
            auto arg = op->MakeArgumentPointer(
                da.data, db.data, std::array<const void *, 0>{}, de.data, al,
                as, bl, bs, std::array<A5, 0>{}, std::array<A5, 0>{}, el, es,
                A2{q.stride, q.stride}, A2{q.dilation, q.dilation},
                A2{q.pad, q.pad}, A2{q.pad, q.pad}, ew::PassThrough{},
                ew::PassThrough{}, ew::PassThrough{});
            if (!op->IsSupportedArgument(arg.get())) {
              auto bytes = de.bytes(stream);
              require(std::all_of(bytes.begin(), bytes.end(),
                                  [](auto c) { return c == 0xa5; }),
                      "refused output changed");
              da.unchanged(a, stream);
              db.unchanged(b, stream);
              ++refusals;
              continue;
            }
            require(op->GetWorkSpaceSize(arg.get()) == 0,
                    "unexpected scratch requirement");
            op->MakeInvokerPointer()->Run(arg.get(), {stream, false});
            auto actual = de.read(gold.size(), stream);
            for (size_t i = 0; i < gold.size(); ++i) {
              double value = full(actual[i]),
                     rounded = full(to_half(float(gold[i]))),
                     error = std::abs(value - gold[i]);
              double spacing = std::max(
                  std::ldexp(1., -24),
                  std::ldexp(1., gold[i] == 0
                                     ? -24
                                     : std::ilogb(std::abs(gold[i])) - 10));
              double tolerance = .5 * spacing + 2e-5 + 5e-5 * std::abs(gold[i]);
              require(std::isfinite(value) &&
                          (mode != 1 ? value == rounded : error <= tolerance),
                      "FP16 CPU-reference mismatch: " + name + "/" +
                          op->GetTypeString());
              max_error = std::max(max_error, error);
              max_rounded_error =
                  std::max(max_rounded_error, std::abs(value - rounded));
            }
            da.unchanged(a, stream);
            db.unchanged(b, stream);
            ++admitted;
            ++executions;
            if (mode == 2)
              ++accumulator_executions;
            counts[name + "/" + std::to_string(ordinal) + "/" +
                   op->GetTypeString()]++;
          }
          require(admitted > 0, "no FP16 instance admitted for fixture");
        }
#endif
}
int main() {
  try {
    hipDeviceProp_t props{};
    check(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName) == "gfx1013:xnack-",
            "native architecture");
    require(!ck::is_xdl_supported(), "unexpected XDL admission");
#ifdef DL_KERNELS
    Buffer<unsigned> flags(3);
    features<<<1, 1>>>(flags.data);
    check(hipGetLastError());
    auto values = flags.read(3, {});
    require(values[0] == 0 && values[1] == 1 && values[2] == 32,
            "native scalar feature predicates");
    result["device_flags"] = {
        {"dot2", values[0]}, {"mac_f32", values[1]}, {"wave_size", values[2]}};
#endif
    run<true>();
    run<false>();
    result["status"] = "passed";
    result["architecture"] = props.gcnArchName;
    result["fixtures"] = fixtures;
    result["executions"] = executions;
    result["refusals"] = refusals;
    result["accumulator_executions"] = accumulator_executions;
    result["kernel_counts"] = counts;
    result["unique_ids"] = ids.size();
    result["max_abs_error"] = max_error;
    result["max_rounded_abs_error"] = max_rounded_error;
    result["guards"] = true;
    std::cout << result.dump() << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
