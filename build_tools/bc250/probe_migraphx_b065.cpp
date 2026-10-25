// SPDX-License-Identifier: MIT
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <iterator>
#include <map>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/onnx.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = migraphx;
namespace fs = std::filesystem;

void expect(bool ok, const std::string &message) {
  if (not ok)
    throw std::runtime_error(message);
}

std::string read(const fs::path &file) {
  std::ifstream stream(file, std::ios::binary);
  expect(bool(stream), "Missing fixture: " + file.string());
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

template <class T> std::vector<T> values(const fs::path &file) {
  auto bytes = read(file);
  expect(bytes.size() % sizeof(T) == 0, "Bad oracle size");
  std::vector<T> result(bytes.size() / sizeof(T));
  std::copy(bytes.begin(), bytes.end(),
            reinterpret_cast<char *>(result.data()));
  return result;
}

int main(int argc, char **argv) {
  try {
    expect(argc == 2, "Usage: probe MODEL_DIRECTORY");
    fs::path root{argv[1]};
    hipDeviceProp_t props{};
    expect(hipGetDeviceProperties(&props, 0) == hipSuccess and
               std::string(props.gcnArchName).rfind("gfx1013:", 0) == 0 and
               props.warpSize == 32,
           "Requires native gfx1013 wave32");
    unsigned passed = 1;
    expect(mx::gpu::gfx_default_rocblas() and
               not mx::gpu::hipblaslt_supported(),
           "Expected rocBLAS-only candidate");
    ++passed;
    for (const std::string name :
         {"pointwise", "reduce", "dense", "transpose", "conv"}) {
      auto folder = root / name;
      auto bytes = read(folder / "model.onnx");
      auto p = mx::parse_onnx((folder / "model.onnx").string());
      const std::map<std::string, std::vector<std::size_t>> expected_lens = {
          {"pointwise", {3, 17}},
          {"reduce", {3, 1}},
          {"dense", {3, 5}},
          {"transpose", {3, 5}},
          {"conv", {2, 3, 3, 3}}};
      auto output_shapes = p.get_output_shapes();
      expect(output_shapes.size() == 1 and
                 output_shapes.front().type() == mx::shape::float_type and
                 output_shapes.front().lens() == expected_lens.at(name),
             "Parsed output shape/type mismatch");
      auto from_buffer = mx::parse_onnx_buffer(bytes.data(), bytes.size(), {});
      expect(p == from_buffer, name + " file/buffer graph mismatch");
      const auto shapes = p.get_parameter_shapes();
      expect(not shapes.empty(), "No input parameters");
      ++passed;
      mx::compile_options options;
      options.offload_copy = true;
      options.fast_math = false;
      options.exhaustive_tune = false;
      p.compile(mx::gpu::target{}, options);
      std::map<std::string, unsigned> operations;
      for (auto ins : mx::iterator_for(*p.get_main_module()))
        ++operations[ins->name()];
      for (const auto &op : operations)
        std::cerr << op.first << ':' << op.second << ' ';
      std::cerr << '\n';
      const std::string required = name == "conv" ? "gpu::convolution"
                                   : (name == "dense" or name == "transpose")
                                       ? "gpu::gemm"
                                       : "gpu::code_object";
      expect(operations.count(required) != 0,
             name + " expected GPU lowering missing");
      for (int replay : {0, 1}) {
        std::map<std::string, std::vector<float>> inputs;
        mx::parameter_map params;
        for (const auto &parameter : shapes) {
          expect(parameter.second.type() == mx::shape::float_type and
                     parameter.second.standard(),
                 "Fixture admits only standard FP32 inputs");
          auto input =
              values<float>(folder / (parameter.first + '-' +
                                      std::to_string(replay) + ".bin"));
          expect(input.size() == parameter.second.elements(),
                 "Input extent mismatch");
          input.insert(input.begin(), 19);
          input.push_back(23);
          inputs.emplace(parameter.first, std::move(input));
          params.emplace(parameter.first,
                         mx::argument(parameter.second,
                                      inputs.at(parameter.first).data() + 1));
        }
        const auto originals = inputs;
        auto expected =
            values<double>(folder / ("cpu-" + std::to_string(replay) + ".f64"));
        auto ort =
            values<float>(folder / ("ort-" + std::to_string(replay) + ".f32"));
        auto result = p.eval(params);
        p.finish();
        expect(result.size() == 1 and
                   result.front().get_shape().type() ==
                       mx::shape::float_type and
                   result.front().get_shape().lens() == expected_lens.at(name),
               "Runtime output shape/type mismatch");
        std::vector<double> actual;
        result.front().visit(
            [&](auto view) { actual.assign(view.begin(), view.end()); });
        expect(actual.size() == expected.size() and
                   ort.size() == expected.size(),
               "Output extent mismatch");
        for (std::size_t i = 0; i < actual.size(); ++i) {
          double tolerance = 2e-5 + 2e-5 * std::abs(expected[i]);
          expect(std::isfinite(actual[i]) and
                     std::abs(actual[i] - expected[i]) <= tolerance and
                     std::abs(actual[i] - double(ort[i])) <= tolerance,
                 name + " CPU-double/CPU-ORT mismatch");
        }
        expect(inputs == originals, name + " host input/guard changed");
        ++passed;
      }
    }
    for (const std::string name : {"unknown", "corrupt"}) {
      for (bool buffer : {false, true}) {
        bool refused = false;
        try {
          auto file = root / (name + ".onnx");
          if (buffer) {
            auto bytes = read(file);
            (void)mx::parse_onnx_buffer(bytes.data(), bytes.size(), {});
          } else
            (void)mx::parse_onnx(file.string());
        } catch (const std::exception &e) {
          const std::string needle = name == "unknown"
                                         ? "Unknown operator: BC250Unknown"
                                         : "Failed reading onnx file";
          expect(std::string(e.what()).find(needle) != std::string::npos,
                 e.what());
          refused = true;
        }
        expect(refused, name + " unexpectedly accepted");
        ++passed;
      }
    }
    std::cout << "{\"status\":\"passed\",\"checks\":" << passed << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
