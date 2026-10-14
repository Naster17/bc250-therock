// Native hipTensor scalar contractions: independent Einstein-summation
// reference. Build with hipcc -O2 --offload-arch=gfx1013 -lhiptensor.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

static void hipCheck(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
static void tensorCheck(hiptensorStatus_t status) {
  if (status != HIPTENSOR_STATUS_SUCCESS)
    throw std::runtime_error(hiptensorGetErrorString(status));
}
struct Layout {
  std::vector<int32_t> modes;
  std::vector<int64_t> lengths, strides;
  size_t size() const {
    size_t result = 1;
    for (size_t i = 0; i < lengths.size(); ++i)
      result += (lengths[i] - 1) * strides[i];
    return result;
  }
  size_t offset(std::map<int32_t, int64_t> const &coordinates) const {
    size_t result = 0;
    for (size_t i = 0; i < modes.size(); ++i)
      result += coordinates.at(modes[i]) * strides[i];
    return result;
  }
};
struct Case {
  std::string name;
  Layout a, b, c, d;
  bool hasC = true, inPlace = false;
  double alpha = 1.25, beta = -0.5;
  bool positiveInputs = false;
  hiptensorOperator_t opA = HIPTENSOR_OP_IDENTITY, opB = HIPTENSOR_OP_IDENTITY,
                      opC = HIPTENSOR_OP_IDENTITY;
};
static double unary(double value, hiptensorOperator_t op) {
  if (op == HIPTENSOR_OP_NEG)
    return -value;
  if (op == HIPTENSOR_OP_ABS)
    return std::abs(value);
  if (op == HIPTENSOR_OP_RELU)
    return std::max(0.0, value);
  if (op == HIPTENSOR_OP_SQRT)
    return sqrt(value);
  if (op == HIPTENSOR_OP_RCP)
    return 1.0 / value;
  if (op == HIPTENSOR_OP_SIGMOID)
    return value >= 0.0 ? 1.0 / (1.0 + exp(-value))
                        : exp(value) / (1.0 + exp(value));
  if (op == HIPTENSOR_OP_TANH)
    return tanh(value);
  if (op == HIPTENSOR_OP_EXP)
    return exp(value);
  if (op == HIPTENSOR_OP_LOG)
    return log(value);
  if (op == HIPTENSOR_OP_SIN)
    return sin(value);
  if (op == HIPTENSOR_OP_COS)
    return cos(value);
  if (op == HIPTENSOR_OP_TAN)
    return tan(value);
  if (op == HIPTENSOR_OP_SINH)
    return sinh(value);
  if (op == HIPTENSOR_OP_COSH)
    return cosh(value);
  if (op == HIPTENSOR_OP_ASIN)
    return asin(value);
  if (op == HIPTENSOR_OP_ACOS)
    return acos(value);
  if (op == HIPTENSOR_OP_ATAN)
    return atan(value);
  if (op == HIPTENSOR_OP_ASINH)
    return asinh(value);
  if (op == HIPTENSOR_OP_ACOSH)
    return acosh(value);
  if (op == HIPTENSOR_OP_ATANH)
    return atanh(value);
  if (op == HIPTENSOR_OP_CEIL)
    return ceil(value);
  if (op == HIPTENSOR_OP_FLOOR)
    return floor(value);
  return value;
}
static hiptensorTensorDescriptor_t descriptor(hiptensorHandle_t handle,
                                              Layout const &layout,
                                              hiptensorDataType_t type) {
  hiptensorTensorDescriptor_t result = nullptr;
  tensorCheck(hiptensorCreateTensorDescriptor(
      handle, &result, layout.modes.size(), layout.lengths.data(),
      layout.strides.data(), type, 1));
  return result;
}
template <class T, class Output = T>
static void
run(hiptensorHandle_t handle, Case const &test, hiptensorDataType_t dtype,
    hiptensorDataType_t outputType = static_cast<hiptensorDataType_t>(-1)) {
  if (outputType == static_cast<hiptensorDataType_t>(-1))
    outputType = dtype;
  using Acc = std::conditional_t<std::is_same_v<Output, double>, double, float>;
  constexpr bool fp64 = std::is_same_v<Output, double>;
  auto compute = fp64 ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
  std::vector<T> a(test.a.size()), b(test.b.size());
  std::vector<Output> c(test.c.size()), d(test.d.size(), Output(-17.0f));
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = T(Acc(int(i % 13) - 6) / Acc(9));
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = T(Acc(int(i % 11) - 5) / Acc(11));
  for (size_t i = 0; i < c.size(); ++i)
    c[i] = Output(Acc(int(i % 7) - 3) / Acc(7));
  if (test.positiveInputs) {
    for (auto &v : a)
      v = T(std::abs(double(v)) + 1.125);
    for (auto &v : b)
      v = T(std::abs(double(v)) + 1.125);
    for (auto &v : c)
      v = Output(std::abs(double(v)) + 1.125);
  }
  if (test.alpha == 0)
    std::fill(a.begin(), a.end(), T(std::numeric_limits<float>::quiet_NaN()));
  if (test.beta == 0)
    std::fill(c.begin(), c.end(),
              Output(std::numeric_limits<float>::quiet_NaN()));
  if (test.inPlace)
    d = c;
  std::vector<Output> expected = d;
  std::vector<bool> written(d.size(), false);
  std::map<int32_t, int64_t> lengths, coordinates;
  for (auto const *layout : {&test.a, &test.b, &test.d})
    for (size_t i = 0; i < layout->modes.size(); ++i)
      lengths[layout->modes[i]] = layout->lengths[i];
  // Enumerate the union of all modes, independent of the GPU's flattened
  // mapping.
  std::vector<int32_t> free = test.d.modes, summed;
  for (auto const &pair : lengths)
    if (std::find(free.begin(), free.end(), pair.first) == free.end())
      summed.push_back(pair.first);
  auto enumerate = [&](std::vector<int32_t> const &modes, auto &&callback) {
    uint64_t count = 1;
    for (auto m : modes)
      count *= lengths.at(m);
    for (uint64_t linear = 0; linear < count; ++linear) {
      uint64_t remainder = linear;
      for (auto m : modes) {
        coordinates[m] = remainder % lengths.at(m);
        remainder /= lengths.at(m);
      }
      callback();
    }
  };
  enumerate(free, [&] {
    double sum = 0;
    if (test.alpha != 0)
      enumerate(summed, [&] {
        sum += unary(double(a[test.a.offset(coordinates)]), test.opA) *
               unary(double(b[test.b.offset(coordinates)]), test.opB);
      });
    double value = test.alpha * sum;
    if (test.hasC && test.beta != 0)
      value +=
          test.beta * unary(double(c[test.c.offset(coordinates)]), test.opC);
    expected[test.d.offset(coordinates)] = Output(value);
    written[test.d.offset(coordinates)] = true;
  });
  auto da = descriptor(handle, test.a, dtype),
       db = descriptor(handle, test.b, dtype),
       dc = test.hasC ? descriptor(handle, test.c, outputType) : nullptr,
       dd = descriptor(handle, test.d, outputType);
  hiptensorOperationDescriptor_t op = nullptr;
  tensorCheck(hiptensorCreateContraction(
      handle, &op, da, test.a.modes.data(), test.opA, db, test.b.modes.data(),
      test.opB, dc, test.hasC ? test.c.modes.data() : nullptr, test.opC, dd,
      test.d.modes.data(), compute));
  hiptensorPlanPreference_t pref = nullptr;
  tensorCheck(hiptensorCreatePlanPreference(
      handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));
  uint64_t workspace = 999;
  tensorCheck(hiptensorEstimateWorkspaceSize(
      handle, op, pref, HIPTENSOR_WORKSPACE_DEFAULT, &workspace));
  if (workspace != 0)
    throw std::runtime_error("scalar workspace must be zero");
  hiptensorPlan_t plan = nullptr;
  tensorCheck(hiptensorCreatePlan(handle, &plan, op, pref, 0));
  // A plan must remain valid after destroying all source
  // descriptors/preferences.
  tensorCheck(hiptensorDestroyOperationDescriptor(op));
  tensorCheck(hiptensorDestroyPlanPreference(pref));
  for (auto desc : {da, db, dc, dd})
    if (desc)
      tensorCheck(hiptensorDestroyTensorDescriptor(desc));
  T *ga = nullptr, *gb = nullptr;
  Output *gc = nullptr, *gd = nullptr;
  hipCheck(hipMalloc(&ga, a.size() * sizeof(T)));
  hipCheck(hipMalloc(&gb, b.size() * sizeof(T)));
  hipCheck(hipMalloc(&gd, d.size() * sizeof(Output)));
  if (test.inPlace)
    gc = gd;
  else
    hipCheck(hipMalloc(&gc, c.size() * sizeof(Output)));
  hipCheck(
      hipMemcpy(ga, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice));
  hipCheck(
      hipMemcpy(gb, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(gc, c.data(), c.size() * sizeof(Output),
                     hipMemcpyHostToDevice));
  if (!test.inPlace)
    hipCheck(hipMemcpy(gd, d.data(), d.size() * sizeof(Output),
                       hipMemcpyHostToDevice));
  hipStream_t stream;
  hipCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  Acc alpha = test.alpha, beta = test.beta;
  tensorCheck(hiptensorContract(
      handle, plan, &alpha, ga, gb, test.hasC ? &beta : nullptr,
      test.hasC ? gc : nullptr, gd, nullptr, 0, stream));
  hipCheck(hipStreamSynchronize(stream));
  hipCheck(hipMemcpy(d.data(), gd, d.size() * sizeof(Output),
                     hipMemcpyDeviceToHost));
  double tolerance =
      fp64 ? 1e-12
           : (std::is_same_v<Output, float>
                  ? 2e-5
                  : (std::is_same_v<Output, __half> ? 0.001 : 0.008));
  for (size_t i = 0; i < d.size(); ++i)
    if ((!written[i] && double(d[i]) != double(expected[i])) ||
        !std::isfinite(double(d[i])) ||
        std::abs(double(d[i]) - double(expected[i])) >
            tolerance * (1 + std::abs(double(expected[i]))))
      throw std::runtime_error(test.name + " mismatch at storage index " +
                               std::to_string(i));
  tensorCheck(hiptensorDestroyPlan(plan));
  hipCheck(hipStreamDestroy(stream));
  hipCheck(hipFree(ga));
  hipCheck(hipFree(gb));
  if (!test.inPlace)
    hipCheck(hipFree(gc));
  hipCheck(hipFree(gd));
  std::cout << "PASS " << test.name << " dtype=" << int(dtype)
            << " output=" << int(outputType) << '\n';
}

static void
reject(hiptensorHandle_t handle, Case const &test, hiptensorStatus_t expected,
       hiptensorOperator_t opA = HIPTENSOR_OP_IDENTITY,
       hiptensorDataType_t btype = HIPTENSOR_R_32F,
       hiptensorComputeDescriptor_t compute = HIPTENSOR_COMPUTE_DESC_32F,
       hiptensorDataType_t atype = HIPTENSOR_R_32F,
       hiptensorDataType_t ctype = HIPTENSOR_R_32F,
       hiptensorDataType_t dtype = HIPTENSOR_R_32F) {
  auto da = descriptor(handle, test.a, atype),
       db = descriptor(handle, test.b, btype),
       dc = descriptor(handle, test.c, ctype),
       dd = descriptor(handle, test.d, dtype);
  hiptensorOperationDescriptor_t op = nullptr;
  auto result = hiptensorCreateContraction(
      handle, &op, da, test.a.modes.data(), opA, db, test.b.modes.data(),
      HIPTENSOR_OP_IDENTITY, dc, test.c.modes.data(), HIPTENSOR_OP_IDENTITY, dd,
      test.d.modes.data(), compute);
  if (result != expected || op != nullptr)
    throw std::runtime_error("expected rejection: " + test.name);
  for (auto desc : {da, db, dc, dd})
    tensorCheck(hiptensorDestroyTensorDescriptor(desc));
  std::cout << "PASS reject " << test.name << '\n';
}
int main() {
  try {
    int device;
    hipCheck(hipGetDevice(&device));
    hipDeviceProp_t props;
    hipCheck(hipGetDeviceProperties(&props, device));
    if (std::string(props.gcnArchName).substr(0, 7) != "gfx1013")
      throw std::runtime_error("requires native gfx1013");
    hiptensorHandle_t handle;
    tensorCheck(hiptensorCreate(&handle));
    Case gemm{"gemm-padded-permuted",
              {{'k', 'm'}, {7, 5}, {1, 11}},
              {{'n', 'k'}, {3, 7}, {2, 13}},
              {{'m', 'n'}, {5, 3}, {9, 1}},
              {{'n', 'm'}, {3, 5}, {1, 7}}};
    Case batch{"batch-multiple-contracted-modes",
               {{'q', 'k', 'm', 'b'}, {2, 3, 5, 2}, {1, 4, 17, 101}},
               {{'n', 'b', 'q', 'k'}, {3, 2, 2, 3}, {1, 97, 5, 19}},
               {{'m', 'n', 'b'}, {5, 3, 2}, {13, 2, 89}},
               {{'b', 'n', 'm'}, {2, 3, 5}, {101, 1, 7}}};
    Case outer{"outer-product",
               {{'m'}, {5}, {2}},
               {{'n'}, {3}, {3}},
               {{'n', 'm'}, {3, 5}, {1, 7}},
               {{'m', 'n'}, {5, 3}, {7, 1}}};
    Case dot{"dot-unit-output-mode",
             {{'k', 'u'}, {257, 1}, {2, 1027}},
             {{'u', 'k'}, {1, 257}, {1029, 3}},
             {{'u'}, {1}, {1}},
             {{'u'}, {1}, {1}}};
    Case threeSums{"three-contracted-modes",
                   {{'q', 'k', 'r', 'm'}, {2, 3, 2, 3}, {1, 3, 13, 31}},
                   {{'r', 'n', 'k', 'q'}, {2, 4, 3, 2}, {1, 3, 17, 61}},
                   {{'m', 'n'}, {3, 4}, {11, 2}},
                   {{'n', 'm'}, {4, 3}, {1, 7}}};
    auto scale = gemm;
    scale.name = "scale-no-c";
    scale.hasC = false;
    scale.beta = 0;
    auto unaryCase = batch;
    unaryCase.name = "unary-neg-abs-relu";
    unaryCase.opA = HIPTENSOR_OP_NEG;
    unaryCase.opB = HIPTENSOR_OP_ABS;
    unaryCase.opC = HIPTENSOR_OP_RELU;
    auto inplace = gemm;
    inplace.name = "in-place-c-d";
    inplace.c = inplace.d;
    inplace.inPlace = true;
    auto alphaZero = gemm;
    alphaZero.name = "alpha-zero-nan-a";
    alphaZero.alpha = 0;
    auto betaZero = gemm;
    betaZero.name = "beta-zero-nan-c";
    betaZero.beta = 0;
    auto largeK = gemm;
    largeK.name = "odd-long-contracted-dimension";
    largeK.a.lengths[0] = largeK.b.lengths[1] = 257;
    largeK.a.strides[1] = 263;
    for (auto const &test : {gemm, batch, outer, scale, unaryCase, inplace,
                             alphaZero, betaZero, largeK, dot, threeSums}) {
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half>(handle, test, HIPTENSOR_R_16F);
      run<hip_bfloat16>(handle, test, HIPTENSOR_R_16BF);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
      run<float, double>(handle, test, HIPTENSOR_R_32F, HIPTENSOR_R_64F);
    }
    {
      auto test = batch;
      test.name = "unary-sqrt";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_SQRT;
      test.positiveInputs = true;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-rcp";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_RCP;
      test.positiveInputs = true;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-sigmoid";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_SIGMOID;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-tanh";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_TANH;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-exp";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_EXP;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-log";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_LOG;
      test.positiveInputs = true;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-sin";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_SIN;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-cos";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_COS;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-tan";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_TAN;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-sinh";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_SINH;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-cosh";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_COSH;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-asin";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_ASIN;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-acos";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_ACOS;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-atan";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_ATAN;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-asinh";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_ASINH;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-acosh";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_ACOSH;
      test.positiveInputs = true;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-atanh";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_ATANH;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-ceil";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_CEIL;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    {
      auto test = batch;
      test.name = "unary-floor";
      test.opA = test.opB = test.opC = HIPTENSOR_OP_FLOOR;
      test.positiveInputs = false;
      run<float>(handle, test, HIPTENSOR_R_32F);
      run<double>(handle, test, HIPTENSOR_R_64F);
      run<__half, float>(handle, test, HIPTENSOR_R_16F, HIPTENSOR_R_32F);
      run<hip_bfloat16, float>(handle, test, HIPTENSOR_R_16BF, HIPTENSOR_R_32F);
    }
    auto duplicate = gemm;
    duplicate.name = "duplicate-mode";
    duplicate.a.modes[1] = 'k';
    reject(handle, duplicate, HIPTENSOR_STATUS_INVALID_VALUE);
    auto length = gemm;
    length.name = "inconsistent-length";
    length.b.lengths[1] = 8;
    reject(handle, length, HIPTENSOR_STATUS_INVALID_VALUE);
    auto overlapping = gemm;
    overlapping.name = "overlapping-output";
    overlapping.d.strides = {1, 1};
    reject(handle, overlapping, HIPTENSOR_STATUS_NOT_SUPPORTED);
    auto unilateral = gemm;
    unilateral.name = "unilateral-reduction";
    unilateral.a.modes[0] = 'z';
    reject(handle, unilateral, HIPTENSOR_STATUS_NOT_SUPPORTED);
    auto unsupported = gemm;
    unsupported.name = "unsupported-unary-complex-conj";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_CONJ);
    unsupported.name = "mixed-precision-tensors";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_IDENTITY, HIPTENSOR_R_64F);
    unsupported.name = "complex-tensor";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_IDENTITY, HIPTENSOR_C_32F);
    unsupported.name = "reduced-precision-accumulation";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_IDENTITY, HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_16F);
    unsupported.name = "mismatched-low-precision-inputs";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_IDENTITY, HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F,
           HIPTENSOR_R_16F);
    unsupported.name = "mismatched-c-d-dtypes";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_IDENTITY, HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F,
           HIPTENSOR_R_16F, HIPTENSOR_R_16F);
    unsupported.name = "unsupported-f16-to-f64";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_IDENTITY, HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_64F,
           HIPTENSOR_R_16F, HIPTENSOR_R_64F, HIPTENSOR_R_64F);
    unsupported.name = "unsupported-f32-to-f16";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_IDENTITY, HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F,
           HIPTENSOR_R_32F, HIPTENSOR_R_16F, HIPTENSOR_R_16F);
    unsupported.name = "unsupported-f64-accumulation-with-f32-output";
    reject(handle, unsupported, HIPTENSOR_STATUS_NOT_SUPPORTED,
           HIPTENSOR_OP_IDENTITY, HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_64F);
    tensorCheck(hiptensorDestroy(handle));
    std::cout << "HIPTENSOR_SCALAR_PASS cases=166\n";
  } catch (std::exception const &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
