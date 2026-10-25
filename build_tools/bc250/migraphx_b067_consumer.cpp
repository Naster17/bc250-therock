// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <migraphx/migraphx.hpp>
#include <stdexcept>
#include <string>
#include <vector>

template <class T> std::vector<T> read_values(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (not f)
    throw std::runtime_error("Missing fixture: " + path);
  auto size = static_cast<std::size_t>(f.tellg());
  if (size % sizeof(T) != 0)
    throw std::runtime_error("Bad fixture length");
  std::vector<T> out(size / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(out.data()), size);
  if (not f)
    throw std::runtime_error("Incomplete fixture");
  return out;
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("Expected models directory");
    const std::string root = argv[1];
    int passed = 0;
    for (const std::string name : {"pointwise", "constant"}) {
      auto p =
          migraphx::parse_onnx((root + "/" + name + "/model.onnx").c_str());
      migraphx::compile_options options;
      options.set_fast_math(false);
      options.set_offload_copy(true);
      options.set_exhaustive_tune_flag(false);
      p.compile(migraphx::target("gpu"), options);
      for (int replay : {0, 1}) {
        const auto folder = root + "/" + name + "/";
        const auto suffix = std::to_string(replay);
        auto x = read_values<float>(folder + "x-" + suffix + ".bin");
        auto bias = name == "pointwise"
                        ? read_values<float>(folder + "b-" + suffix + ".bin")
                        : std::vector<float>{};
        const auto original_x = x;
        const auto original_bias = bias;
        const auto shapes = p.get_parameter_shapes();
        migraphx::program_parameters params;
        params.add("x", migraphx::argument(shapes["x"], x.data()));
        if (name == "pointwise")
          params.add("b", migraphx::argument(shapes["b"], bias.data()));
        const auto outputs = p.eval(params);
        auto actual = outputs[0].as_vector<float>();
        auto reference = read_values<double>(folder + "cpu-" + suffix + ".f64");
        auto ort = read_values<float>(folder + "ort-" + suffix + ".f32");
        if (actual.size() != reference.size() or actual.size() != ort.size())
          throw std::runtime_error("Output size mismatch");
        for (std::size_t i = 0; i < actual.size(); ++i)
          if (not std::isfinite(actual[i]) or
              std::abs(actual[i] - reference[i]) >
                  2e-5 + 2e-5 * std::abs(reference[i]) or
              std::abs(actual[i] - ort[i]) > 2e-5 + 2e-5 * std::abs(ort[i]))
            throw std::runtime_error("Reference mismatch");
        if (x != original_x or bias != original_bias)
          throw std::runtime_error("Host input modified");
        ++passed;
      }
    }
    std::cout << "{\"status\":\"passed\",\"checks\":" << passed << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
