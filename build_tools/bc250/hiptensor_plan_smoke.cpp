// Public plan API contracts and independent GPU reference checks on gfx1013.
#define main bc250RealContractionMain
#include "hiptensor_scalar_smoke.cpp"
#undef main
#include <complex>
#include <cstring>
#include <hip/hip_complex.h>
using Reference = std::complex<double>;
static Reference complexUnary(Reference v, hiptensorOperator_t op) {
  return op == HIPTENSOR_OP_CONJ ? std::conj(v) : v;
}
template <class T> static T storage(Reference v) {
  T result{};
  result.x = v.real();
  result.y = v.imag();
  return result;
}
template <class T> static Reference reference(T v) { return {v.x, v.y}; }
static int planChecks = 0;
static void expect(hiptensorStatus_t actual, hiptensorStatus_t expected,
                   const char *label) {
  if (actual != expected)
    throw std::runtime_error(std::string(label) + ": " +
                             hiptensorGetErrorString(actual));
  ++planChecks;
  std::cout << "PASS plan API " << label << " status=" << int(actual) << '\n';
}
static void apiChecks(hiptensorHandle_t handle) {
  hiptensorPlanPreference_t pref = nullptr;
  for (auto algo : {HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT,
                    HIPTENSOR_ALGO_ACTOR_CRITIC}) {
    expect(hiptensorCreatePlanPreference(handle, &pref, algo,
                                         HIPTENSOR_JIT_MODE_NONE),
           HIPTENSOR_STATUS_SUCCESS, "valid algo");
    tensorCheck(hiptensorDestroyPlanPreference(pref));
  }
  for (int value : {0, 1, -2, -99, 999}) {
    pref = nullptr;
    expect(hiptensorCreatePlanPreference(handle, &pref,
                                         static_cast<hiptensorAlgo_t>(value),
                                         HIPTENSOR_JIT_MODE_NONE),
           HIPTENSOR_STATUS_INVALID_VALUE, "invalid algo");
    if (pref)
      throw std::runtime_error("failed preference not null");
  }
  expect(hiptensorCreatePlanPreference(handle, &pref, HIPTENSOR_ALGO_DEFAULT,
                                       HIPTENSOR_JIT_MODE_DEFAULT),
         HIPTENSOR_STATUS_NOT_SUPPORTED, "JIT unsupported");
  expect(hiptensorCreatePlanPreference(handle, &pref, HIPTENSOR_ALGO_DEFAULT,
                                       static_cast<hiptensorJitMode_t>(99)),
         HIPTENSOR_STATUS_INVALID_VALUE, "invalid JIT");
  expect(hiptensorCreatePlanPreference(nullptr, &pref, HIPTENSOR_ALGO_DEFAULT,
                                       HIPTENSOR_JIT_MODE_NONE),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "null handle");
  expect(hiptensorCreatePlanPreference(handle, nullptr, HIPTENSOR_ALGO_DEFAULT,
                                       HIPTENSOR_JIT_MODE_NONE),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "null output");
  tensorCheck(hiptensorCreatePlanPreference(
      handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));
  int32_t value = 0;
  for (int attr = 0; attr < 6; ++attr) {
    auto attribute = static_cast<hiptensorPlanPreferenceAttribute_t>(attr);
    for (size_t size :
         {size_t(0), size_t(1), size_t(3), size_t(5), size_t(8), size_t(4096)})
      expect(hiptensorPlanPreferenceSetAttribute(handle, pref, attribute,
                                                 &value, size),
             HIPTENSOR_STATUS_INVALID_VALUE, "bad attribute size");
    expect(hiptensorPlanPreferenceSetAttribute(handle, pref, attribute, nullptr,
                                               sizeof(value)),
           HIPTENSOR_STATUS_NOT_INITIALIZED, "null attribute buffer");
  }
  expect(hiptensorPlanPreferenceSetAttribute(nullptr, pref,
                                             HIPTENSOR_PLAN_PREFERENCE_JIT,
                                             &value, sizeof(value)),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "null setter handle");
  expect(hiptensorPlanPreferenceSetAttribute(handle, nullptr,
                                             HIPTENSOR_PLAN_PREFERENCE_JIT,
                                             &value, sizeof(value)),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "null setter pref");
  expect(hiptensorPlanPreferenceSetAttribute(
             handle, pref, static_cast<hiptensorPlanPreferenceAttribute_t>(99),
             &value, sizeof(value)),
         HIPTENSOR_STATUS_NOT_SUPPORTED, "unknown attribute");
  for (auto attr :
       {HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE,
        HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, HIPTENSOR_PLAN_PREFERENCE_JIT}) {
    value = 1;
    expect(hiptensorPlanPreferenceSetAttribute(handle, pref, attr, &value,
                                               sizeof(value)),
           HIPTENSOR_STATUS_NOT_SUPPORTED, "unsupported request");
    for (int invalid : {-1, 2, 99})
      expect(hiptensorPlanPreferenceSetAttribute(handle, pref, attr, &invalid,
                                                 sizeof(invalid)),
             HIPTENSOR_STATUS_INVALID_VALUE, "invalid enum");
    value = 0;
    expect(hiptensorPlanPreferenceSetAttribute(handle, pref, attr, &value,
                                               sizeof(value)),
           HIPTENSOR_STATUS_SUCCESS, "supported request");
  }
  for (auto attr : {HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT,
                    HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK}) {
    for (int invalid : {-1, -99})
      expect(hiptensorPlanPreferenceSetAttribute(handle, pref, attr, &invalid,
                                                 sizeof(invalid)),
             HIPTENSOR_STATUS_INVALID_VALUE, "negative count/rank");
    for (int unsupported : {1, 7, 2147483647})
      expect(hiptensorPlanPreferenceSetAttribute(
                 handle, pref, attr, &unsupported, sizeof(unsupported)),
             HIPTENSOR_STATUS_NOT_SUPPORTED, "unsupported count/rank");
    value = 0;
    expect(hiptensorPlanPreferenceSetAttribute(handle, pref, attr, &value,
                                               sizeof(value)),
           HIPTENSOR_STATUS_SUCCESS, "zero count/rank");
  }
  for (auto algo : {HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT,
                    HIPTENSOR_ALGO_ACTOR_CRITIC})
    expect(hiptensorPlanPreferenceSetAttribute(handle, pref,
                                               HIPTENSOR_PLAN_PREFERENCE_ALGO,
                                               &algo, sizeof(algo)),
           HIPTENSOR_STATUS_SUCCESS, "set valid algo");
  value = 99;
  expect(hiptensorPlanPreferenceSetAttribute(handle, pref,
                                             HIPTENSOR_PLAN_PREFERENCE_ALGO,
                                             &value, sizeof(value)),
         HIPTENSOR_STATUS_INVALID_VALUE, "set invalid algo");
  tensorCheck(hiptensorDestroyPlanPreference(pref));
  std::cout << "PLAN_API_VALIDATION_COMPLETE\n";
}
template <class T> static T element(Reference v) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    return T(v.real());
  else
    return storage<T>(v);
}
template <class T> static Reference asReference(T value) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    return {double(value), 0};
  else
    return reference(value);
}
template <class T> static void numerical(hiptensorHandle_t handle) {
  constexpr bool real = std::is_same_v<T, float> || std::is_same_v<T, double>;
  constexpr bool wide =
      std::is_same_v<T, double> || std::is_same_v<T, hipDoubleComplex>;
  auto dtype = real ? (wide ? HIPTENSOR_R_64F : HIPTENSOR_R_32F)
                    : (wide ? HIPTENSOR_C_64F : HIPTENSOR_C_32F);
  auto compute =
      real ? (wide ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F)
           : (wide ? HIPTENSOR_COMPUTE_DESC_C64F : HIPTENSOR_COMPUTE_DESC_C32F);
  Layout a{{'m', 'k'}, {2, 3}, {1, 2}}, b{{'k', 'n'}, {3, 2}, {1, 3}},
      d{{'m', 'n'}, {2, 2}, {1, 2}};
  auto da = descriptor(handle, a, dtype), db = descriptor(handle, b, dtype),
       dd = descriptor(handle, d, dtype);
  hiptensorOperationDescriptor_t op = nullptr;
  tensorCheck(hiptensorCreateContraction(
      handle, &op, da, a.modes.data(),
      real ? HIPTENSOR_OP_IDENTITY : HIPTENSOR_OP_CONJ, db, b.modes.data(),
      HIPTENSOR_OP_NEG, nullptr, nullptr, HIPTENSOR_OP_IDENTITY, dd,
      d.modes.data(), compute));
  hiptensorPlanPreference_t pref = nullptr;
  tensorCheck(hiptensorCreatePlanPreference(
      handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));
  uint64_t estimate = 321;
  for (auto preference : {HIPTENSOR_WORKSPACE_MIN, HIPTENSOR_WORKSPACE_DEFAULT,
                          HIPTENSOR_WORKSPACE_MAX}) {
    expect(
        hiptensorEstimateWorkspaceSize(handle, op, pref, preference, &estimate),
        HIPTENSOR_STATUS_SUCCESS, "workspace");
    if (estimate)
      throw std::runtime_error("nonzero scalar workspace");
  }
  estimate = 321;
  expect(hiptensorEstimateWorkspaceSize(
             handle, op, pref, static_cast<hiptensorWorksizePreference_t>(99),
             &estimate),
         HIPTENSOR_STATUS_INVALID_VALUE, "bad workspace preference");
  if (estimate != 321)
    throw std::runtime_error("failed query changed output");
  expect(hiptensorEstimateWorkspaceSize(nullptr, op, pref,
                                        HIPTENSOR_WORKSPACE_DEFAULT, &estimate),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "query handle");
  expect(hiptensorEstimateWorkspaceSize(handle, nullptr, pref,
                                        HIPTENSOR_WORKSPACE_DEFAULT, &estimate),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "query op");
  expect(hiptensorEstimateWorkspaceSize(handle, op, nullptr,
                                        HIPTENSOR_WORKSPACE_DEFAULT, &estimate),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "query pref");
  expect(hiptensorEstimateWorkspaceSize(handle, op, pref,
                                        HIPTENSOR_WORKSPACE_DEFAULT, nullptr),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "query output");
  hiptensorPlan_t failed = nullptr;
  expect(hiptensorCreatePlan(nullptr, &failed, op, pref, 0),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "plan handle");
  expect(hiptensorCreatePlan(handle, nullptr, op, pref, 0),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "plan output");
  expect(hiptensorCreatePlan(handle, &failed, nullptr, pref, 0),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "plan op");
  expect(hiptensorCreatePlan(handle, &failed, op, nullptr, 0),
         HIPTENSOR_STATUS_NOT_INITIALIZED, "plan pref");
  if (failed)
    throw std::runtime_error("failed plan not null");
  std::vector<hiptensorPlan_t> plans;
  uint64_t limit = 0;
  for (auto algo : {HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT,
                    HIPTENSOR_ALGO_ACTOR_CRITIC}) {
    tensorCheck(hiptensorPlanPreferenceSetAttribute(
        handle, pref, HIPTENSOR_PLAN_PREFERENCE_ALGO, &algo, sizeof(algo)));
    // Failed setters must leave the preference usable and existing plans
    // intact.
    int32_t invalid = 99, unsupported = 1;
    expect(hiptensorPlanPreferenceSetAttribute(handle, pref,
                                               HIPTENSOR_PLAN_PREFERENCE_ALGO,
                                               &invalid, sizeof(invalid)),
           HIPTENSOR_STATUS_INVALID_VALUE, "atomic invalid setter");
    expect(hiptensorPlanPreferenceSetAttribute(
               handle, pref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &unsupported,
               sizeof(unsupported)),
           HIPTENSOR_STATUS_NOT_SUPPORTED, "atomic unsupported setter");
    hiptensorPlan_t plan = nullptr;
    expect(hiptensorCreatePlan(handle, &plan, op, pref, limit),
           HIPTENSOR_STATUS_SUCCESS, "create owned plan");
    plans.push_back(plan);
    limit = limit == 0 ? 1 : std::numeric_limits<uint64_t>::max();
    struct {
      uint64_t before, value, after;
    } guarded{123, 321, 456};
    expect(hiptensorPlanGetAttribute(handle, plan,
                                     HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                     &guarded.value, sizeof(guarded.value)),
           HIPTENSOR_STATUS_SUCCESS, "plan workspace");
    if (guarded.before != 123 || guarded.after != 456 || guarded.value != 0)
      throw std::runtime_error("workspace guard");
    guarded.value = 321;
    for (size_t size :
         {size_t(0), size_t(1), size_t(7), size_t(9), size_t(16), size_t(4096)})
      expect(hiptensorPlanGetAttribute(handle, plan,
                                       HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                       &guarded.value, size),
             HIPTENSOR_STATUS_INVALID_VALUE, "getter size");
    expect(hiptensorPlanGetAttribute(handle, plan,
                                     static_cast<hiptensorPlanAttribute_t>(99),
                                     &guarded.value, 8),
           HIPTENSOR_STATUS_NOT_SUPPORTED, "getter attribute");
    expect(hiptensorPlanGetAttribute(nullptr, plan,
                                     HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                     &guarded.value, 8),
           HIPTENSOR_STATUS_NOT_INITIALIZED, "getter handle");
    expect(hiptensorPlanGetAttribute(handle, nullptr,
                                     HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                     &guarded.value, 8),
           HIPTENSOR_STATUS_NOT_INITIALIZED, "getter plan");
    expect(hiptensorPlanGetAttribute(
               handle, plan, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, nullptr, 8),
           HIPTENSOR_STATUS_NOT_INITIALIZED, "getter buffer");
    if (guarded.before != 123 || guarded.after != 456 || guarded.value != 321)
      throw std::runtime_error("failed getter writes output");
  }
  tensorCheck(hiptensorDestroyPlanPreference(pref));
  tensorCheck(hiptensorDestroyOperationDescriptor(op));
  for (auto desc : {da, db, dd})
    tensorCheck(hiptensorDestroyTensorDescriptor(desc));
  std::vector<T> av(6), bv(6), dv(8, element<T>({-17, 13}));
  for (int i = 0; i < 6; ++i) {
    av[i] = element<T>({double(i - 2) / 4, double(i % 3 - 1) / 8});
    bv[i] = element<T>({double(3 - i) / 5, double(i % 2) / 7});
  }
  auto alpha = element<T>({1.25, -0.375});
  std::vector<Reference> ref(4);
  for (int n = 0; n < 2; ++n)
    for (int m = 0; m < 2; ++m)
      for (int k = 0; k < 3; ++k)
        ref[m + 2 * n] +=
            complexUnary(asReference(av[m + 2 * k]),
                         real ? HIPTENSOR_OP_IDENTITY : HIPTENSOR_OP_CONJ) *
            (-asReference(bv[k + 3 * n])) * asReference(alpha);
  T *ga, *gb, *gd;
  hipCheck(hipMalloc(&ga, 6 * sizeof(T)));
  hipCheck(hipMalloc(&gb, 6 * sizeof(T)));
  hipCheck(hipMalloc(&gd, 8 * sizeof(T)));
  hipCheck(hipMemcpy(ga, av.data(), 6 * sizeof(T), hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(gb, bv.data(), 6 * sizeof(T), hipMemcpyHostToDevice));
  hipStream_t stream;
  hipCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  for (auto plan : plans) {
    hipCheck(hipMemcpy(gd, dv.data(), 8 * sizeof(T), hipMemcpyHostToDevice));
    tensorCheck(hiptensorContract(handle, plan, &alpha, ga, gb, nullptr,
                                  nullptr, gd + 2, nullptr, 0, stream));
    hipCheck(hipStreamSynchronize(stream));
    std::vector<T> actual(8);
    hipCheck(
        hipMemcpy(actual.data(), gd, 8 * sizeof(T), hipMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i)
      if (i < 2 || i >= 6) {
        if (std::memcmp(&actual[i], &dv[i], sizeof(T)))
          throw std::runtime_error("output guard");
      } else if (!std::isfinite(std::abs(asReference(actual[i]))) ||
                 std::abs(asReference(actual[i]) - ref[i - 2]) >
                     (wide ? 1e-12 : 2e-5) * (1 + std::abs(ref[i - 2])))
        throw std::runtime_error("plan numerical mismatch");
    ++planChecks;
    std::cout << "PASS plan GPU dtype=" << int(dtype) << '\n';
    tensorCheck(hiptensorDestroyPlan(plan));
  }
  hipCheck(hipStreamDestroy(stream));
  hipCheck(hipFree(ga));
  hipCheck(hipFree(gb));
  hipCheck(hipFree(gd));
}
int main(int argc, char **argv) {
  try {
    hipDeviceProp_t props;
    hipCheck(hipGetDeviceProperties(&props, 0));
    if (std::string(props.gcnArchName).substr(0, 7) != "gfx1013")
      throw std::runtime_error("requires gfx1013");
    hiptensorHandle_t handle;
    tensorCheck(hiptensorCreate(&handle));
    if (argc > 1 && std::string(argv[1]) == "--baseline") {
      hiptensorPlanPreference_t pref = nullptr;
      auto status = hiptensorCreatePlanPreference(
          handle, &pref, static_cast<hiptensorAlgo_t>(99),
          HIPTENSOR_JIT_MODE_NONE);
      if (pref)
        tensorCheck(hiptensorDestroyPlanPreference(pref));
      tensorCheck(hiptensorDestroy(handle));
      if (status != HIPTENSOR_STATUS_SUCCESS)
        throw std::runtime_error("baseline no longer admits invalid algo");
      std::cout << "BASELINE_ACCEPTED_INVALID_ALGO\n";
      return 0;
    }
    apiChecks(handle);
    numerical<float>(handle);
    numerical<double>(handle);
    numerical<hipFloatComplex>(handle);
    numerical<hipDoubleComplex>(handle);
    tensorCheck(hiptensorDestroy(handle));
    std::cout << "HIPTENSOR_PLAN_PASS cases=" << planChecks << '\n';
  } catch (std::exception const &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
