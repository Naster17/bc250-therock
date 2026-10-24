// SPDX-License-Identifier: MIT
// Small FP32 graph admission fixtures; no model/performance/autograd claim.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <algorithm>
#include <cmath>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <map>
#include <migraphx/compile_options.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = migraphx;

void expect(bool value, const std::string &message) {
  if (not value)
    throw std::runtime_error(message);
}

void compile(mx::program &p, const std::string &required) {
  mx::compile_options options;
  options.offload_copy = true;
  options.fast_math = false;
  options.exhaustive_tune = false;
  p.compile(mx::gpu::target{}, options);
  std::map<std::string, int> ops;
  for (auto ins : mx::iterator_for(*p.get_main_module()))
    ++ops[ins->name()];
  for (const auto &op : ops)
    std::cerr << op.first << ':' << op.second << ' ';
  std::cerr << '\n';
  bool found = ops.count(required) != 0;
  if (required == "gpu::gemm")
    found = found or ops.count("gpu::hip_gemm") != 0;
  expect(found, "Expected GPU lowering absent: " + required);
}

void compare(mx::program &p, const mx::parameter_map &params,
             const std::vector<double> &reference, const std::string &label) {
  auto outputs = p.eval(params);
  p.finish();
  expect(outputs.size() == 1, "Unexpected output count");
  std::vector<double> values;
  outputs.front().visit([&](auto v) { values.assign(v.begin(), v.end()); });
  expect(values.size() == reference.size(), "Wrong output size");
  for (std::size_t i = 0; i < values.size(); ++i)
    expect(std::isfinite(values[i]) and
               std::abs(values[i] - reference[i]) <=
                   2e-5 + 2e-5 * std::abs(reference[i]),
           label + " CPU-double mismatch");
}

int main() {
  try {
    hipDeviceProp_t props{};
    expect(hipGetDeviceProperties(&props, 0) == hipSuccess,
           "HIP device query failed");
    expect(std::string(props.gcnArchName).rfind("gfx1013:", 0) == 0 and
               props.warpSize == 32,
           "Native gfx1013 wave32 required");
    unsigned passed = 1;
#if MIGRAPHX_USE_HIPBLASLT
#error The B06.4 fixture qualifies the rocBLAS-only build
#endif
    mx::gpu::context ctx;
    expect(mx::gpu::gfx_default_rocblas() and
               mx::gpu::gfx_default_rocblas(ctx) and
               not mx::gpu::hipblaslt_supported() and
               not mx::gpu::hipblaslt_supported(ctx),
           "rocBLAS-only provider query contract failed");
    ++passed;
    // Strided activations and broadcast row bias, both signs before ReLU.
    for (std::size_t width : {17u, 33u}) {
      const std::size_t stride = width + 5;
      mx::shape sx{mx::shape::float_type, {3, width}, {stride, 1}};
      mx::shape sb{mx::shape::float_type, {3, 1}};
      mx::program p;
      auto *m = p.get_main_module();
      auto x = m->add_parameter("x", sx);
      auto b = m->add_parameter("b", sb);
      auto bc = m->add_instruction(
          mx::make_op("multibroadcast", {{"out_lens", {3, width}}}), b);
      auto add = m->add_instruction(mx::make_op("add"), x, bc);
      auto relu = m->add_instruction(mx::make_op("relu"), add);
      m->add_return({relu});
      compile(p, "gpu::code_object");
      for (int replay : {0, 1}) {
        std::vector<float> input(sx.element_space() + 2, 19),
            bias{23, -0.25f, 0, 0.25f, 23};
        std::vector<double> ref;
        for (std::size_t row = 0; row < 3; ++row)
          for (std::size_t col = 0; col < width; ++col) {
            float value = float(int((row * 7 + col + replay) % 31) - 15) / 16;
            input[1 + row * stride + col] = value;
            ref.push_back(std::max(0.0, double(value) + double(bias[row + 1])));
          }
        auto original = input, saved_bias = bias;
        compare(p,
                {{"x", mx::argument(sx, input.data() + 1)},
                 {"b", mx::argument(sb, bias.data() + 1)}},
                ref, "pointwise");
        expect(input == original and bias == saved_bias,
               "Pointwise input/guard changed");
        ++passed;
      }
    }
    for (std::size_t width : {17u, 33u, 129u, 513u}) {
      mx::shape s{mx::shape::float_type, {3, width}};
      mx::program p;
      auto *m = p.get_main_module();
      auto x = m->add_parameter("x", s);
      auto r =
          m->add_instruction(mx::make_op("reduce_sum", {{"axes", {1}}}), x);
      m->add_return({r});
      compile(p, "gpu::code_object");
      for (int replay : {0, 1}) {
        std::vector<float> input(s.elements() + 2, 19);
        std::vector<double> ref(3, 0);
        for (std::size_t row = 0; row < 3; ++row)
          for (std::size_t col = 0; col < width; ++col) {
            float value = float(int((row * 7 + col + replay) % 23) - 11) / 16;
            input[1 + row * width + col] = value;
            ref[row] += double(value);
          }
        auto original = input;
        compare(p, {{"x", mx::argument(s, input.data() + 1)}}, ref, "reduce");
        expect(input == original, "Reduction input/guard changed");
        ++passed;
      }
    }
    for (std::size_t inner : {7u, 17u}) {
      mx::shape sa{mx::shape::float_type, {3, inner}};
      mx::shape sb{mx::shape::float_type, {inner, 5}};
      mx::program p;
      auto *m = p.get_main_module();
      auto a = m->add_parameter("a", sa);
      auto b = m->add_parameter("b", sb);
      auto dot = m->add_instruction(mx::make_op("dot"), a, b);
      m->add_return({dot});
      compile(p, "gpu::gemm");
      for (int replay : {0, 1}) {
        std::vector<float> av(sa.elements() + 2, 19), bv(sb.elements() + 2, 23);
        for (std::size_t i = 0; i < sa.elements(); ++i)
          av[1 + i] = float(int((i + replay) % 13) - 6) / 16;
        for (std::size_t i = 0; i < sb.elements(); ++i)
          bv[1 + i] = float(int((i + replay) % 17) - 8) / 16;
        std::vector<double> ref(15, 0);
        for (std::size_t row = 0; row < 3; ++row)
          for (std::size_t col = 0; col < 5; ++col)
            for (std::size_t k = 0; k < inner; ++k)
              ref[row * 5 + col] +=
                  double(av[1 + row * inner + k]) * double(bv[1 + k * 5 + col]);
        auto saved_a = av, saved_b = bv;
        compare(p,
                {{"a", mx::argument(sa, av.data() + 1)},
                 {"b", mx::argument(sb, bv.data() + 1)}},
                ref, "gemm");
        expect(av == saved_a and bv == saved_b, "GEMM input/guard changed");
        ++passed;
      }
    }
    for (std::size_t batch : {1u, 2u}) {
      mx::shape sx{mx::shape::float_type, {batch, 2, 5, 5}};
      mx::shape sw{mx::shape::float_type, {3, 2, 3, 3}};
      mx::program p;
      auto *m = p.get_main_module();
      auto x = m->add_parameter("x", sx);
      auto w = m->add_parameter("w", sw);
      auto conv =
          m->add_instruction(mx::make_op("convolution", {{"padding", {0, 0}},
                                                         {"stride", {1, 1}},
                                                         {"dilation", {1, 1}},
                                                         {"group", 1}}),
                             x, w);
      m->add_return({conv});
      compile(p, "gpu::convolution");
      for (int replay : {0, 1}) {
        std::vector<float> xv(sx.elements() + 2, 19), wv(sw.elements() + 2, 23);
        for (std::size_t i = 0; i < sx.elements(); ++i)
          xv[1 + i] = float(int((i + replay) % 17) - 8) / 16;
        for (std::size_t i = 0; i < sw.elements(); ++i)
          wv[1 + i] = float(int((i + replay) % 13) - 6) / 16;
        std::vector<double> ref(batch * 3 * 3 * 3, 0);
        for (std::size_t b = 0; b < batch; ++b)
          for (std::size_t o = 0; o < 3; ++o)
            for (std::size_t r = 0; r < 3; ++r)
              for (std::size_t c = 0; c < 3; ++c)
                for (std::size_t i = 0; i < 2; ++i)
                  for (std::size_t kr = 0; kr < 3; ++kr)
                    for (std::size_t kc = 0; kc < 3; ++kc)
                      ref[((b * 3 + o) * 3 + r) * 3 + c] +=
                          double(
                              xv[1 + ((b * 2 + i) * 5 + r + kr) * 5 + c + kc]) *
                          double(wv[1 + ((o * 2 + i) * 3 + kr) * 3 + kc]);
        auto saved_x = xv, saved_w = wv;
        compare(p,
                {{"x", mx::argument(sx, xv.data() + 1)},
                 {"w", mx::argument(sw, wv.data() + 1)}},
                ref, "convolution");
        expect(xv == saved_x and wv == saved_w, "Conv input/guard changed");
        ++passed;
      }
    }
    std::cout << "{\"status\":\"passed\",\"checks\":" << passed << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
