// Native gfx1013 complex contractions; independent std::complex<double>
// reference. Reuse public-API layout/descriptor helpers and retain a separately
// callable real regression suite. No private library headers or CPU fallback
// are used.
#define main bc250RealContractionMain
#include "hiptensor_scalar_smoke.cpp"
#undef main
#include <complex>
#include <cstring>
#include <hip/hip_complex.h>

using Reference = std::complex<double>;
struct ComplexCase {
  Case shape;
  Reference alpha = {1.25, -0.375}, beta = {-0.5, 0.25};
};
static int complexCases = 0;
static Reference complexUnary(Reference value, hiptensorOperator_t op) {
  if (op == HIPTENSOR_OP_NEG)
    return -value;
  if (op == HIPTENSOR_OP_CONJ)
    return std::conj(value);
  return value;
}
template <class T> static T storage(Reference v) {
  T result{};
  result.x = v.real();
  result.y = v.imag();
  return result;
}
template <class T> static Reference reference(T v) { return {v.x, v.y}; }

template <class T>
static void runComplex(hiptensorHandle_t handle, ComplexCase test) {
  constexpr bool fp64 = std::is_same_v<T, hipDoubleComplex>;
  auto dtype = fp64 ? HIPTENSOR_C_64F : HIPTENSOR_C_32F;
  auto compute =
      fp64 ? HIPTENSOR_COMPUTE_DESC_C64F : HIPTENSOR_COMPUTE_DESC_C32F;
  auto const &s = test.shape;
  T guard = storage<T>({-17.0, 13.0});
  std::vector<T> a(s.a.size() + 4, guard), b(s.b.size() + 4, guard),
      c(s.c.size() + 4, guard), d(s.d.size() + 4, guard);
  for (size_t i = 0; i < s.a.size(); ++i)
    a[i + 2] =
        storage<T>({double(int(i % 13) - 6) / 9, double(int(i % 7) - 3) / 8});
  for (size_t i = 0; i < s.b.size(); ++i)
    b[i + 2] =
        storage<T>({double(int(i % 11) - 5) / 11, double(int(i % 5) - 2) / 6});
  for (size_t i = 0; i < s.c.size(); ++i)
    c[i + 2] =
        storage<T>({double(int(i % 7) - 3) / 7, double(int(i % 9) - 4) / 10});
  if (test.alpha == Reference{})
    for (auto *input : {&a, &b})
      std::fill(input->begin() + 2, input->end() - 2, storage<T>({NAN, NAN}));
  if (test.beta == Reference{})
    std::fill(c.begin() + 2, c.end() - 2, storage<T>({NAN, NAN}));
  if (s.inPlace)
    d = c;
  auto expected = d;
  std::vector<bool> written(d.size(), false);
  std::map<int32_t, int64_t> lengths, coordinates;
  for (auto const *layout : {&s.a, &s.b, &s.d})
    for (size_t i = 0; i < layout->modes.size(); ++i)
      lengths[layout->modes[i]] = layout->lengths[i];
  std::vector<int32_t> summed;
  for (auto [mode, len] : lengths)
    if (std::find(s.d.modes.begin(), s.d.modes.end(), mode) == s.d.modes.end())
      summed.push_back(mode);
  auto enumerate = [&](auto const &modes, auto callback) {
    uint64_t count = 1;
    for (auto m : modes)
      count *= lengths.at(m);
    for (uint64_t linear = 0; linear < count; ++linear) {
      auto remainder = linear;
      for (auto m : modes) {
        coordinates[m] = remainder % lengths.at(m);
        remainder /= lengths.at(m);
      }
      callback();
    }
  };
  // Round the public scalar storage before evaluating the independent
  // reference.
  T alpha = storage<T>(test.alpha), beta = storage<T>(test.beta);
  enumerate(s.d.modes, [&] {
    Reference sum{};
    if (reference(alpha) != Reference{})
      enumerate(summed, [&] {
        sum += complexUnary(reference(a[s.a.offset(coordinates) + 2]), s.opA) *
               complexUnary(reference(b[s.b.offset(coordinates) + 2]), s.opB);
      });
    auto value = reference(alpha) * sum;
    if (s.hasC && reference(beta) != Reference{})
      value += reference(beta) *
               complexUnary(reference(c[s.c.offset(coordinates) + 2]), s.opC);
    auto offset = s.d.offset(coordinates) + 2;
    expected[offset] = storage<T>(value);
    written[offset] = true;
  });
  auto da = descriptor(handle, s.a, dtype), db = descriptor(handle, s.b, dtype),
       dc = s.hasC ? descriptor(handle, s.c, dtype) : nullptr,
       dd = descriptor(handle, s.d, dtype);
  hiptensorOperationDescriptor_t op = nullptr;
  tensorCheck(hiptensorCreateContraction(handle, &op, da, s.a.modes.data(),
                                         s.opA, db, s.b.modes.data(), s.opB, dc,
                                         s.hasC ? s.c.modes.data() : nullptr,
                                         s.opC, dd, s.d.modes.data(), compute));
  hiptensorPlanPreference_t pref = nullptr;
  tensorCheck(hiptensorCreatePlanPreference(handle, &pref, s.algo, s.jit));
  uint64_t workspace = 999;
  tensorCheck(hiptensorEstimateWorkspaceSize(
      handle, op, pref, s.workspacePreference, &workspace));
  if (workspace != 0)
    throw std::runtime_error("complex scalar workspace must be zero");
  hiptensorPlan_t plan = nullptr;
  tensorCheck(hiptensorCreatePlan(handle, &plan, op, pref, 0));
  tensorCheck(hiptensorDestroyOperationDescriptor(op));
  tensorCheck(hiptensorDestroyPlanPreference(pref));
  for (auto desc : {da, db, dc, dd})
    if (desc)
      tensorCheck(hiptensorDestroyTensorDescriptor(desc));
  T *ga, *gb, *gc, *gd;
  hipCheck(hipMalloc(&ga, a.size() * sizeof(T)));
  hipCheck(hipMalloc(&gb, b.size() * sizeof(T)));
  hipCheck(hipMalloc(&gd, d.size() * sizeof(T)));
  if (s.inPlace)
    gc = gd;
  else
    hipCheck(hipMalloc(&gc, c.size() * sizeof(T)));
  hipCheck(
      hipMemcpy(ga, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice));
  hipCheck(
      hipMemcpy(gb, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice));
  hipCheck(
      hipMemcpy(gc, c.data(), c.size() * sizeof(T), hipMemcpyHostToDevice));
  if (!s.inPlace)
    hipCheck(
        hipMemcpy(gd, d.data(), d.size() * sizeof(T), hipMemcpyHostToDevice));
  hipStream_t stream;
  hipCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  auto contract = [&](const void *aPtr, const void *bPtr, const void *cPtr,
                      void *dPtr) {
    return hiptensorContract(handle, plan, &alpha, aPtr, bPtr,
                             s.hasC ? &beta : nullptr, s.hasC ? cPtr : nullptr,
                             dPtr, nullptr, 0, stream);
  };
  for (void *alias : {static_cast<void *>(ga + 2), static_cast<void *>(ga + 3),
                      static_cast<void *>(gb + 2)})
    if (contract(ga + 2, gb + 2, gc + 2, alias) !=
        HIPTENSOR_STATUS_NOT_SUPPORTED)
      throw std::runtime_error("complex input/output alias accepted");
  if (s.hasC && s.d.size() > 1 &&
      contract(ga + 2, gb + 2, gd + 3, gd + 2) !=
          HIPTENSOR_STATUS_NOT_SUPPORTED)
    throw std::runtime_error("complex partial C/D alias accepted");
  if (contract(nullptr, gb + 2, gc + 2, gd + 2) !=
      HIPTENSOR_STATUS_INVALID_VALUE)
    throw std::runtime_error("complex null A accepted");
  if (s.hasC && reference(beta) != Reference{} &&
      contract(ga + 2, gb + 2, nullptr, gd + 2) !=
          HIPTENSOR_STATUS_INVALID_VALUE)
    throw std::runtime_error("complex missing nonzero-beta C accepted");
  tensorCheck(contract(
      ga + 2, gb + 2,
      (s.hasC && reference(beta) == Reference{}) ? nullptr : gc + 2, gd + 2));
  hipCheck(hipStreamSynchronize(stream));
  hipCheck(
      hipMemcpy(d.data(), gd, d.size() * sizeof(T), hipMemcpyDeviceToHost));
  double tolerance = fp64 ? 1e-12 : 2e-5;
  for (size_t i = 0; i < d.size(); ++i) {
    if (!written[i]) {
      if (std::memcmp(&d[i], &expected[i], sizeof(T)))
        throw std::runtime_error(s.name + " guard mismatch");
    } else if (!std::isfinite(d[i].x) || !std::isfinite(d[i].y) ||
               std::abs(reference(d[i]) - reference(expected[i])) >
                   tolerance * (1 + std::abs(reference(expected[i]))))
      throw std::runtime_error(s.name + " complex mismatch at " +
                               std::to_string(i));
  }
  // Verify input buffers and their surrounding guards were not modified.
  for (auto pair : {std::make_pair(ga, &a), std::make_pair(gb, &b)}) {
    std::vector<T> actual(pair.second->size());
    hipCheck(hipMemcpy(actual.data(), pair.first, actual.size() * sizeof(T),
                       hipMemcpyDeviceToHost));
    if (std::memcmp(actual.data(), pair.second->data(),
                    actual.size() * sizeof(T)))
      throw std::runtime_error(s.name + " input modified");
  }
  if (s.measure) {
    hipEvent_t start, stop;
    hipCheck(hipEventCreate(&start));
    hipCheck(hipEventCreate(&stop));
    hipCheck(hipEventRecord(start, stream));
    for (int i = 0; i < 30; ++i)
      tensorCheck(contract(ga + 2, gb + 2, gc + 2, gd + 2));
    hipCheck(hipEventRecord(stop, stream));
    hipCheck(hipEventSynchronize(stop));
    float ms;
    hipCheck(hipEventElapsedTime(&ms, start, stop));
    std::cout << "TIMING " << s.name << " dtype=" << int(dtype)
              << " output=" << int(dtype)
              << " repeats=30 mean_us=" << ms * 1000 / 30 << '\n';
    hipCheck(hipEventDestroy(start));
    hipCheck(hipEventDestroy(stop));
  }
  hipCheck(hipStreamDestroy(stream));
  tensorCheck(hiptensorDestroyPlan(plan));
  hipCheck(hipFree(ga));
  hipCheck(hipFree(gb));
  if (!s.inPlace)
    hipCheck(hipFree(gc));
  hipCheck(hipFree(gd));
  std::cout << "PASS complex " << s.name << " dtype=" << int(dtype) << '\n';
  ++complexCases;
}

static void rejectComplex(
    hiptensorHandle_t handle, Case shape,
    hiptensorDataType_t atype = HIPTENSOR_C_32F,
    hiptensorDataType_t btype = HIPTENSOR_C_32F,
    hiptensorDataType_t ctype = HIPTENSOR_C_32F,
    hiptensorDataType_t dtype = HIPTENSOR_C_32F,
    hiptensorComputeDescriptor_t compute = HIPTENSOR_COMPUTE_DESC_C32F,
    hiptensorStatus_t expected = HIPTENSOR_STATUS_NOT_SUPPORTED) {
  auto da = descriptor(handle, shape.a, atype),
       db = descriptor(handle, shape.b, btype),
       dc = descriptor(handle, shape.c, ctype),
       dd = descriptor(handle, shape.d, dtype);
  hiptensorOperationDescriptor_t op = nullptr;
  auto result = hiptensorCreateContraction(
      handle, &op, da, shape.a.modes.data(), shape.opA, db,
      shape.b.modes.data(), shape.opB, dc, shape.c.modes.data(), shape.opC, dd,
      shape.d.modes.data(), compute);
  if (result != expected || op)
    throw std::runtime_error("expected complex rejection: " + shape.name);
  for (auto desc : {da, db, dc, dd})
    tensorCheck(hiptensorDestroyTensorDescriptor(desc));
  std::cout << "PASS reject complex " << shape.name << '\n';
  ++complexCases;
}
int main() {
  try {
    hipDeviceProp_t props;
    hipCheck(hipGetDeviceProperties(&props, 0));
    if (std::string(props.gcnArchName).substr(0, 7) != "gfx1013")
      throw std::runtime_error("requires native gfx1013");
    std::cout << "NATIVE_ARCH " << props.gcnArchName << '\n';
    hiptensorHandle_t handle;
    tensorCheck(hiptensorCreate(&handle));
    Case gemm{"gemm-padded-permuted",
              {{'k', 'm'}, {7, 5}, {1, 11}},
              {{'n', 'k'}, {3, 7}, {2, 13}},
              {{'m', 'n'}, {5, 3}, {9, 1}},
              {{'n', 'm'}, {3, 5}, {1, 7}}};
    Case mixed{"unilateral-shared-batch",
               {{'z', 'k', 'm', 'b'}, {2, 3, 5, 2}, {1, 4, 17, 101}},
               {{'n', 'b', 'k', 'r'}, {3, 2, 3, 2}, {1, 97, 5, 19}},
               {{'m', 'n', 'b'}, {5, 3, 2}, {13, 2, 89}},
               {{'b', 'n', 'm'}, {2, 3, 5}, {101, 1, 7}}};
    Case outer{
        "outer-product", {{'m'}, {5}, {2}}, {{'n'}, {3}, {3}}, gemm.c, gemm.d};
    Case dot{"dot-unit-output",
             {{'k', 'u'}, {257, 1}, {2, 1027}},
             {{'u', 'k'}, {1, 257}, {1029, 3}},
             {{'u'}, {1}, {1}},
             {{'u'}, {1}, {1}}};
    auto aOnly = gemm;
    aOnly.name = "unilateral-a-only";
    aOnly.b = {{'n'}, {3}, {3}};
    auto bOnly = gemm;
    bOnly.name = "unilateral-b-only";
    bOnly.a = {{'m'}, {5}, {2}};
    auto independent = gemm;
    independent.name = "independent-sums";
    independent.a.modes[0] = 'z';
    for (auto base : {gemm, mixed, outer, dot, aOnly, bOnly, independent}) {
      for (auto opA :
           {HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_NEG, HIPTENSOR_OP_CONJ})
        for (auto opB :
             {HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_NEG, HIPTENSOR_OP_CONJ})
          for (auto opC :
               {HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_NEG, HIPTENSOR_OP_CONJ}) {
            auto shape = base;
            shape.opA = opA;
            shape.opB = opB;
            shape.opC = opC;
            shape.name += "-ops-" + std::to_string(int(opA)) + "-" +
                          std::to_string(int(opB)) + "-" +
                          std::to_string(int(opC));
            runComplex<hipFloatComplex>(handle, {shape});
            runComplex<hipDoubleComplex>(handle, {shape});
          }
    }
    for (auto shape : {gemm, mixed}) {
      auto noC = shape;
      noC.hasC = false;
      noC.name += "-no-c";
      runComplex<hipFloatComplex>(handle, {noC});
      runComplex<hipDoubleComplex>(handle, {noC});
      auto inplace = shape;
      inplace.name += "-in-place-conj-c";
      inplace.inPlace = true;
      inplace.c = inplace.d;
      inplace.opC = HIPTENSOR_OP_CONJ;
      runComplex<hipFloatComplex>(handle, {inplace});
      runComplex<hipDoubleComplex>(handle, {inplace});
      for (auto scalars :
           {std::make_pair(Reference(-0.0, 0.0), Reference(-0.5, 0.25)),
            std::make_pair(Reference(1.25, -0.375), Reference(0.0, -0.0)),
            std::make_pair(Reference(-0.0, 0.0), Reference(0.0, -0.0)),
            std::make_pair(Reference(0.0, 0.75), Reference(0.0, -0.5))}) {
        shape.name += "-scalars";
        runComplex<hipFloatComplex>(handle,
                                    {shape, scalars.first, scalars.second});
        runComplex<hipDoubleComplex>(handle,
                                     {shape, scalars.first, scalars.second});
      }
    }
    for (auto algo : {HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT,
                      HIPTENSOR_ALGO_ACTOR_CRITIC})
      for (auto preference :
           {HIPTENSOR_WORKSPACE_MIN, HIPTENSOR_WORKSPACE_DEFAULT,
            HIPTENSOR_WORKSPACE_MAX}) {
        auto shape = mixed;
        shape.algo = algo;
        shape.workspacePreference = preference;
        shape.name = "plan-" + std::to_string(int(algo)) + "-" +
                     std::to_string(int(preference));
        runComplex<hipFloatComplex>(handle, {shape});
        runComplex<hipDoubleComplex>(handle, {shape});
      }
    for (auto shape : {gemm, mixed}) {
      shape.measure = true;
      shape.name += "-timed";
      runComplex<hipFloatComplex>(handle, {shape});
      runComplex<hipDoubleComplex>(handle, {shape});
    }
    for (auto op : {HIPTENSOR_OP_ABS, HIPTENSOR_OP_RELU, HIPTENSOR_OP_SQRT,
                    HIPTENSOR_OP_EXP}) {
      for (int operand = 0; operand < 3; ++operand) {
        auto shape = gemm;
        shape.name = "unsupported-op-" + std::to_string(int(op)) + "-" +
                     std::to_string(operand);
        (operand == 0 ? shape.opA : operand == 1 ? shape.opB : shape.opC) = op;
        rejectComplex(handle, shape);
      }
    }
    rejectComplex(handle, gemm, HIPTENSOR_C_32F, HIPTENSOR_C_64F);
    rejectComplex(handle, gemm, HIPTENSOR_C_32F, HIPTENSOR_C_32F,
                  HIPTENSOR_C_64F, HIPTENSOR_C_64F,
                  HIPTENSOR_COMPUTE_DESC_C64F);
    rejectComplex(handle, gemm, HIPTENSOR_C_64F, HIPTENSOR_C_64F,
                  HIPTENSOR_C_32F, HIPTENSOR_C_32F);
    rejectComplex(handle, gemm, HIPTENSOR_R_32F, HIPTENSOR_R_32F);
    rejectComplex(handle, gemm, HIPTENSOR_C_32F, HIPTENSOR_C_32F,
                  HIPTENSOR_C_32F, HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_32F);
    rejectComplex(handle, gemm, HIPTENSOR_C_32F, HIPTENSOR_C_32F,
                  HIPTENSOR_C_64F, HIPTENSOR_C_32F);
    auto bad = gemm;
    bad.name = "complex-span-overflow";
    bad.a.strides[0] = std::numeric_limits<int64_t>::max();
    rejectComplex(handle, bad, HIPTENSOR_C_32F, HIPTENSOR_C_32F,
                  HIPTENSOR_C_32F, HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F,
                  HIPTENSOR_STATUS_INVALID_VALUE);
    bad = gemm;
    bad.name = "complex-overlap-output";
    bad.d.strides = {1, 1};
    rejectComplex(handle, bad);
    bad = gemm;
    bad.name = "complex-mismatched-length";
    bad.b.lengths[1] = 8;
    rejectComplex(handle, bad, HIPTENSOR_C_32F, HIPTENSOR_C_32F,
                  HIPTENSOR_C_32F, HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F,
                  HIPTENSOR_STATUS_INVALID_VALUE);
    auto many = outer;
    many.name = "complex-16-mode-union-with-unit-axes";
    for (int i = 0; i < 14; ++i) {
      auto &layout = i < 7 ? many.a : many.b;
      auto stride = layout.size() + 3;
      layout.modes.push_back(1000 + i);
      layout.lengths.push_back(i % 3 == 0 ? 2 : 1);
      layout.strides.push_back(stride);
    }
    many.opA = HIPTENSOR_OP_CONJ;
    many.opB = HIPTENSOR_OP_NEG;
    runComplex<hipFloatComplex>(handle, {many});
    runComplex<hipDoubleComplex>(handle, {many});
    auto nextStride = many.a.size() + 3;
    many.name = "complex-17-modes-rejected";
    many.a.modes.push_back(2000);
    many.a.lengths.push_back(2);
    many.a.strides.push_back(nextStride);
    rejectComplex(handle, many);
    bad = gemm;
    bad.name = "complex-duplicate-mode";
    bad.a.modes[1] = 'k';
    rejectComplex(handle, bad, HIPTENSOR_C_32F, HIPTENSOR_C_32F,
                  HIPTENSOR_C_32F, HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F,
                  HIPTENSOR_STATUS_INVALID_VALUE);
    bad = gemm;
    bad.name = "complex-zero-input-stride";
    bad.a.strides[0] = 0;
    rejectComplex(handle, bad, HIPTENSOR_C_32F, HIPTENSOR_C_32F,
                  HIPTENSOR_C_32F, HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F,
                  HIPTENSOR_STATUS_INVALID_VALUE);
    tensorCheck(hiptensorDestroy(handle));
    std::cout << "HIPTENSOR_COMPLEX_PASS cases=" << complexCases << '\n';
  } catch (std::exception const &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
