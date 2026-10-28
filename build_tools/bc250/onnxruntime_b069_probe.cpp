// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <map>
#include <migraphx/migraphx.hpp>
#include <stdexcept>
#include <string>
#include <vector>

void hip_check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
template <class T> std::vector<T> read_values(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (not f)
    throw std::runtime_error("Missing fixture: " + path);
  auto bytes = static_cast<std::size_t>(f.tellg());
  if (bytes % sizeof(T) != 0)
    throw std::runtime_error("Invalid fixture size");
  std::vector<T> values(bytes / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(values.data()), bytes);
  if (not f)
    throw std::runtime_error("Incomplete fixture");
  return values;
}
struct device_buffer {
  static constexpr std::size_t guard = 32;
  static constexpr float sentinel = -988.75f;
  std::size_t elements;
  float *base = nullptr;
  explicit device_buffer(std::size_t n) : elements(n) {
    hip_check(hipMalloc(&base, (n + 2 * guard) * sizeof(float)));
    std::vector<float> initial(n + 2 * guard, sentinel);
    hip_check(hipMemcpy(base, initial.data(), initial.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  }
  device_buffer(const device_buffer &) = delete;
  ~device_buffer() { (void)hipFree(base); }
  float *data() { return base + guard; }
  void check_guards() {
    std::vector<float> values(elements + 2 * guard);
    hip_check(hipMemcpy(values.data(), base, values.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    if (not std::all_of(values.begin(), values.begin() + guard,
                        [](float v) { return v == sentinel; }) or
        not std::all_of(values.end() - guard, values.end(),
                        [](float v) { return v == sentinel; }))
      throw std::runtime_error("Device allocation guard modified");
  }
};

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("Expected fixture root");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, 0));
    if (std::string(properties.gcnArchName) != "gfx1013:xnack-")
      throw std::runtime_error("Unexpected native architecture");
    hipStream_t private_stream = nullptr;
    hip_check(hipStreamCreateWithFlags(&private_stream, hipStreamNonBlocking));
    int passed = 0;
    for (const std::string name : {"pointwise", "constant"}) {
      const auto folder = std::string(argv[1]) + "/" + name + "/";
      auto bytes = read_values<char>(folder + "model.onnx");
      migraphx::onnx_options onnx_options;
      onnx_options.set_external_data_path(folder);
      onnx_options.set_input_parameter_shape("x", {3, 17});
      auto program =
          migraphx::parse_onnx_buffer(bytes.data(), bytes.size(), onnx_options);
      migraphx::compile_options options;
      options.set_fast_math(false);
      options.set_offload_copy(false);
      options.set_exhaustive_tune_flag(false);
      program.compile(migraphx::target("gpu"), options);
      auto shapes = program.get_parameter_shapes();
      const auto outputs = program.get_output_shapes();
      if (outputs.size() != 1 or outputs[0].type() != migraphx_shape_float_type)
        throw std::runtime_error("Unexpected output contract");
      for (auto stream : {hipStream_t{}, private_stream}) {
        for (int replay : {0, 1}) {
          const auto suffix = std::to_string(replay);
          auto x = read_values<float>(folder + "x-" + suffix + ".bin");
          auto bias = name == "pointwise"
                          ? read_values<float>(folder + "b-" + suffix + ".bin")
                          : std::vector<float>{};
          const auto x_before = x, bias_before = bias;
          device_buffer input(x.size()), b(bias.size()),
              output(outputs[0].elements());
          hip_check(hipMemcpyAsync(input.data(), x.data(),
                                   x.size() * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
          if (not bias.empty())
            hip_check(hipMemcpyAsync(b.data(), bias.data(),
                                     bias.size() * sizeof(float),
                                     hipMemcpyHostToDevice, stream));
          migraphx::program_parameters params;
          params.add("x", migraphx::argument(shapes["x"], input.data()));
          if (name == "pointwise")
            params.add("b", migraphx::argument(shapes["b"], b.data()));
          int explicit_outputs = 0;
          for (const auto &parameter : shapes.names()) {
            const std::string key = parameter;
            if (key.rfind("main:#output_", 0) == 0) {
              if (shapes[parameter].bytes() != outputs[0].bytes())
                throw std::runtime_error("Output parameter size mismatch");
              params.add(parameter,
                         migraphx::argument(shapes[parameter], output.data()));
              ++explicit_outputs;
            }
          }
          // Match ORT's device-pointer API boundary, not the previous host-copy
          // eval.
          auto result = program.run_async(params, stream);
          if (result.size() != 1 or
              result[0].get_shape().lengths() != outputs[0].lengths())
            throw std::runtime_error("Async output shape mismatch");
          if (explicit_outputs == 0)
            hip_check(hipMemcpyWithStream(output.data(), result[0].data(),
                                          outputs[0].bytes(),
                                          hipMemcpyDeviceToDevice, stream));
          std::vector<float> actual(outputs[0].elements());
          hip_check(hipMemcpyAsync(actual.data(), output.data(),
                                   outputs[0].bytes(), hipMemcpyDeviceToHost,
                                   stream));
          hip_check(hipStreamSynchronize(stream));
          auto cpu = read_values<double>(folder + "cpu-" + suffix + ".f64");
          auto ort = read_values<float>(folder + "ort-" + suffix + ".f32");
          if (actual.size() != cpu.size() or actual.size() != ort.size())
            throw std::runtime_error("Reference shape mismatch");
          for (std::size_t i = 0; i < actual.size(); ++i)
            if (not std::isfinite(actual[i]) or
                std::abs(actual[i] - cpu[i]) > 2e-5 + 2e-5 * std::abs(cpu[i]) or
                std::abs(actual[i] - ort[i]) > 2e-5 + 2e-5 * std::abs(ort[i]))
              throw std::runtime_error("Async reference mismatch");
          input.check_guards();
          b.check_guards();
          output.check_guards();
          std::vector<float> input_after(x.size()), bias_after(bias.size());
          hip_check(hipMemcpy(input_after.data(), input.data(),
                              x.size() * sizeof(float), hipMemcpyDeviceToHost));
          if (not bias.empty())
            hip_check(hipMemcpy(bias_after.data(), b.data(),
                                bias.size() * sizeof(float),
                                hipMemcpyDeviceToHost));
          if (input_after != x_before or bias_after != bias_before or
              x != x_before or bias != bias_before)
            throw std::runtime_error("Input modified");
          ++passed;
        }
      }
    }
    hip_check(hipStreamDestroy(private_stream));
    std::cout << "{\"status\":\"passed\",\"checks\":" << passed
              << ",\"native_architecture\":\"" << properties.gcnArchName
              << "\"}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
