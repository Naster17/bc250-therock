// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <limits>
#include <vector>

static void hip_check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
    std::exit(1);
  }
}
static void lt_check(hipblasStatus_t status,
                     hipblasStatus_t expected = HIPBLAS_STATUS_SUCCESS) {
  if (status != expected) {
    std::fprintf(stderr, "hipBLASLt: status=%d expected=%d\n", int(status),
                 int(expected));
    std::exit(1);
  }
}
struct Layout {
  int rows, cols, ld, stride, batches;
  bool row;
  size_t index(int i, int j, int batch) const {
    return size_t(batch) * stride + (row ? i * ld + j : j * ld + i);
  }
  size_t size() const {
    return size_t(batches - 1) * stride + size_t(row ? rows : cols) * ld;
  }
};
static hipblasLtMatrixLayout_t descriptor(hipDataType type, Layout x) {
  hipblasLtMatrixLayout_t result{};
  lt_check(hipblasLtMatrixLayoutCreate(&result, type, x.rows, x.cols, x.ld));
  const hipblasLtOrder_t order =
      x.row ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
  const int64_t stride = x.stride;
  lt_check(hipblasLtMatrixLayoutSetAttribute(
      result, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  lt_check(hipblasLtMatrixLayoutSetAttribute(
      result, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &x.batches,
      sizeof(x.batches)));
  lt_check(hipblasLtMatrixLayoutSetAttribute(
      result, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride)));
  return result;
}
template <typename T> static T *upload(const std::vector<T> &values) {
  T *result = nullptr;
  hip_check(hipMalloc(&result, values.size() * sizeof(T)));
  hip_check(hipMemcpy(result, values.data(), values.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return result;
}

template <typename Input, typename Output>
static void run(hipblasLtHandle_t handle, hipDataType input_type,
                hipDataType output_type, bool row, bool ta, bool tb,
                int batches, bool zero_beta, bool device_scalars, bool inplace,
                bool implicit, int &passed, int k = 23, bool zero_alpha = false,
                hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT,
                hipDataType bias_type = HIPBLASLT_DATATYPE_INVALID,
                int m = 17, int n = 19, bool benchmark = false) {
  const bool bias_enabled = epilogue == HIPBLASLT_EPILOGUE_BIAS ||
                            epilogue == HIPBLASLT_EPILOGUE_RELU_BIAS ||
                            epilogue == HIPBLASLT_EPILOGUE_GELU_BIAS;
  const bool gelu = epilogue == HIPBLASLT_EPILOGUE_GELU ||
                    epilogue == HIPBLASLT_EPILOGUE_GELU_BIAS;
  const int bias_stride = batches > 1 && tb ? m + 5 : 0;
  std::vector<float> bias_float(size_t(batches - 1) * bias_stride + m, 0);
  std::vector<Output> bias_output(bias_float.size(), Output(0.0f));
  for (int batch = 0; batch < batches; ++batch)
    for (int i = 0; i < m; ++i) {
      const float value = float((i * 3 + batch * 7) % 17 - 8) / 8;
      bias_float[size_t(batch) * bias_stride + i] = value;
      bias_output[size_t(batch) * bias_stride + i] = Output(value);
    }
  auto make = [=](int rows, int cols, bool broadcast) {
    const int ld = (row ? cols : rows) + 3;
    return Layout{rows,    cols,
                  ld,      broadcast ? 0 : (row ? rows : cols) * ld + 7,
                  batches, row};
  };
  const auto a = make(ta ? k : m, ta ? m : k, batches > 1 && ta);
  const auto b = make(tb ? n : k, tb ? k : n, batches > 1 && tb);
  const auto c = make(m, n, batches > 1 && !inplace);
  auto d = make(m, n, false);
  if (!inplace) {
    d.ld += 2;
    d.stride = (row ? m : n) * d.ld + 11;
  }
  std::vector<Input> ha(std::max(size_t(1), a.size()), Input(0.125f)),
      hb(std::max(size_t(1), b.size()), Input(0.125f));
  std::vector<Output> hc(c.size(), Output(0.125f)),
      hd(d.size(), Output(0.125f));
  for (int batch = 0; batch < batches; ++batch) {
    for (int i = 0; i < a.rows; ++i)
      for (int j = 0; j < a.cols; ++j)
        ha[a.index(i, j, batch)] =
            Input(zero_alpha ? std::numeric_limits<float>::quiet_NaN()
                             : float((i * 7 + j * 3 + batch) % 17 - 8) / 16);
    for (int i = 0; i < b.rows; ++i)
      for (int j = 0; j < b.cols; ++j)
        hb[b.index(i, j, batch)] =
            Input(zero_alpha ? std::numeric_limits<float>::quiet_NaN()
                             : float((i * 3 + j * 5 + batch) % 19 - 9) / 16);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        hc[c.index(i, j, batch)] =
            Output(zero_beta ? std::numeric_limits<float>::quiet_NaN()
                             : float((i + j) % 7 - 3) / 8);
  }
  if (inplace)
    hd = hc;
  auto expected = hd;
  std::vector<bool> active(hd.size(), false);
  const float alpha = zero_alpha ? 0.0f : 0.75f,
              beta = zero_beta ? 0.0f : -0.5f;
  for (int batch = 0; batch < batches; ++batch)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        double sum = 0;
        for (int p = 0; p < k && !zero_alpha; ++p)
          sum += double(float(ha[a.index(ta ? p : i, ta ? i : p, batch)])) *
                 double(float(hb[b.index(tb ? j : p, tb ? p : j, batch)]));
        double value = alpha * sum +
                       (zero_beta ? 0 : beta * float(hc[c.index(i, j, batch)]));
        if (bias_enabled) {
          const size_t index = size_t(batch) * bias_stride + i;
          value += bias_type == HIP_R_32F ? bias_float[index]
                                          : float(bias_output[index]);
        }
        if (epilogue == HIPBLASLT_EPILOGUE_RELU ||
            epilogue == HIPBLASLT_EPILOGUE_RELU_BIAS)
          value = std::max(0.0, value);
        if (gelu)
          value = 0.5 * value *
                  (1 + std::tanh(0.7978845608028654 * value *
                                 (1 + 0.044715 * value * value)));
        expected[d.index(i, j, batch)] = Output(float(value));
        active[d.index(i, j, batch)] = true;
      }
  auto da = upload(ha), db = upload(hb);
  auto dc = upload(hc), dd = inplace ? dc : upload(hd);
  const auto ad = descriptor(input_type, a), bd = descriptor(input_type, b);
  const auto cd = descriptor(output_type, c), ddsc = descriptor(output_type, d);
  hipblasLtMatmulDesc_t op{};
  hipblasLtMatmulPreference_t pref{};
  lt_check(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  lt_check(hipblasLtMatmulPreferenceCreate(&pref));
  const auto opa = ta ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const auto opb = tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &opa, sizeof(opa)));
  lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &opb, sizeof(opb)));
  const auto pm = device_scalars ? HIPBLASLT_POINTER_MODE_DEVICE
                                 : HIPBLASLT_POINTER_MODE_HOST;
  lt_check(hipblasLtMatmulDescSetAttribute(
      op, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)));
  lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue)));
  void *bias_gpu = nullptr;
  if (bias_enabled) {
    int empty_count = 9;
    hipblasLtMatmulHeuristicResult_t empty{};
    lt_check(hipblasLtMatmulAlgoGetHeuristic(handle, op, ad, bd, cd, ddsc, pref,
                                             1, &empty, &empty_count),
             HIPBLAS_STATUS_INVALID_VALUE);
    if (empty_count != 0)
      std::exit(1);
    lt_check(hipblasLtMatmul(handle, op, &alpha, da, ad, db, bd, &beta, dc, cd,
                             dd, ddsc, nullptr, nullptr, 0, nullptr),
             HIPBLAS_STATUS_INVALID_VALUE);
    bias_gpu = bias_type == HIP_R_32F
                   ? static_cast<void *>(upload(bias_float))
                   : static_cast<void *>(upload(bias_output));
    lt_check(hipblasLtMatmulDescSetAttribute(
        op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_gpu, sizeof(bias_gpu)));
    if (bias_type != HIPBLASLT_DATATYPE_INVALID)
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
          sizeof(bias_type)));
    lt_check(hipblasLtMatmulDescSetAttribute(
        op, HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &bias_stride,
        sizeof(bias_stride)));
  }
  hipblasLtMatmulHeuristicResult_t result{};
  int count = 0;
  lt_check(hipblasLtMatmulAlgoGetHeuristic(handle, op, ad, bd, cd, ddsc, pref,
                                           1, &result, &count));
  if (count != 1 || result.state != HIPBLAS_STATUS_SUCCESS ||
      result.workspaceSize != 0)
    std::exit(1);
  float *scalars =
      device_scalars ? upload(std::vector<float>{alpha, beta}) : nullptr;
  hipStream_t stream{};
  hip_check(hipStreamCreate(&stream));
  lt_check(hipblasLtMatmul(
      handle, op, scalars ? scalars : &alpha, da, ad, db, bd,
      scalars ? scalars + 1 : &beta,
      zero_beta && !inplace && !device_scalars ? nullptr : dc, cd, dd, ddsc,
      implicit ? nullptr : &result.algo, nullptr, 0, stream));
  hip_check(hipStreamSynchronize(stream));
  hip_check(hipMemcpy(hd.data(), dd, hd.size() * sizeof(Output),
                      hipMemcpyDeviceToHost));
  for (size_t i = 0; i < hd.size(); ++i) {
    const float actual = float(hd[i]), reference = float(expected[i]);
    // Dyadic GEMM/bias/ReLU are exact. GELU uses a CPU double tanh reference;
    // permit FP32 function error and one output rounding interval for
    // FP16/BF16.
    const float rounding = output_type == HIP_R_16BF  ? 0.0078125f
                           : output_type == HIP_R_16F ? 0.0009765625f
                                                      : 0.0f;
    const float tolerance = gelu && active[i]
                                ? 4e-6f * std::max(1.0f, std::fabs(reference)) +
                                      rounding * std::fabs(reference)
                                : 0.0f;
    if (!std::isfinite(actual) || std::fabs(actual - reference) > tolerance) {
      std::fprintf(stderr,
                   "GEMM mismatch case=%d element=%zu got=%.9g expected=%.9g\n",
                   passed, i, actual, reference);
      std::exit(1);
    }
  }
  if (benchmark) {
    // Capture repeated public API launches to measure short kernels without
    // host submission gaps. Compare identical graph fixtures across libraries.
    constexpr int repeats = 32;
    hipGraph_t graph{};
    hipGraphExec_t executable{};
    hip_check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int repeat = 0; repeat < repeats; ++repeat)
      lt_check(hipblasLtMatmul(handle, op, scalars ? scalars : &alpha, da, ad,
                              db, bd, scalars ? scalars + 1 : &beta, dc, cd, dd,
                              ddsc, &result.algo, nullptr, 0, stream));
    hip_check(hipStreamEndCapture(stream, &graph));
    hip_check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int warmup = 0; warmup < 3; ++warmup)
      hip_check(hipGraphLaunch(executable, stream));
    hip_check(hipStreamSynchronize(stream));
    hipEvent_t start{}, stop{};
    hip_check(hipEventCreate(&start));
    hip_check(hipEventCreate(&stop));
    for (int sample = 0; sample < 5; ++sample) {
      hip_check(hipEventRecord(start, stream));
      hip_check(hipGraphLaunch(executable, stream));
      hip_check(hipEventRecord(stop, stream));
      hip_check(hipEventSynchronize(stop));
      float ms = 0;
      hip_check(hipEventElapsedTime(&ms, start, stop));
      std::printf("TIMING m=%d n=%d k=%d input=%d output=%d row=%d ta=%d tb=%d batches=%d sample=%d us=%.6f\n",
                  m,n,k,int(input_type),int(output_type),int(row),int(ta),int(tb),batches,sample,ms*1000/repeats);
    }
    hip_check(hipEventDestroy(start));
    hip_check(hipEventDestroy(stop));
    hip_check(hipGraphExecDestroy(executable));
    hip_check(hipGraphDestroy(graph));
  }
  auto invalid = result.algo;
  invalid.data[0] ^= 1;
  lt_check(hipblasLtMatmul(handle, op, &alpha, da, ad, db, bd, &beta, dc, cd,
                           dd, ddsc, &invalid, nullptr, 0, stream),
           HIPBLAS_STATUS_INVALID_VALUE);
  // Auxiliary GELU is supported now, but a missing auxiliary buffer is invalid.
  const auto missing_aux =
      bias_gpu ? HIPBLASLT_EPILOGUE_GELU_AUX_BIAS : HIPBLASLT_EPILOGUE_GELU_AUX;
  lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &missing_aux, sizeof(missing_aux)));
  count = 9;
  lt_check(hipblasLtMatmulAlgoGetHeuristic(handle, op, ad, bd, cd, ddsc, pref,
                                           1, &result, &count),
           HIPBLAS_STATUS_INVALID_VALUE);
  if (count != 0)
    std::exit(1);
  const auto unsupported = HIPBLASLT_EPILOGUE_BGRADA;
  lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &unsupported, sizeof(unsupported)));
  count = 9;
  lt_check(hipblasLtMatmulAlgoGetHeuristic(handle, op, ad, bd, cd, ddsc, pref,
                                           1, &result, &count),
           HIPBLAS_STATUS_NOT_SUPPORTED);
  if (count != 0)
    std::exit(1);
  hip_check(hipStreamDestroy(stream));
  if (scalars)
    hip_check(hipFree(scalars));
  if (bias_gpu)
    hip_check(hipFree(bias_gpu));
  hip_check(hipFree(da));
  hip_check(hipFree(db));
  hip_check(hipFree(dc));
  if (!inplace)
    hip_check(hipFree(dd));
  for (auto x : {ad, bd, cd, ddsc})
    lt_check(hipblasLtMatrixLayoutDestroy(x));
  lt_check(hipblasLtMatmulPreferenceDestroy(pref));
  lt_check(hipblasLtMatmulDescDestroy(op));
  ++passed;
}

int main() {
  hipblasLtHandle_t handle{};
  lt_check(hipblasLtCreate(&handle));
  int passed = 0;
  for (bool row : {false, true})
    for (bool ta : {false, true})
      for (bool tb : {false, true})
        for (int batches : {1, 3})
          for (bool zero_beta : {false, true}) {
            const bool device = ta != tb, inplace = !zero_beta && !ta;
            run<float, float>(handle, HIP_R_32F, HIP_R_32F, row, ta, tb,
                              batches, zero_beta, device, inplace, row, passed);
            run<__half, __half>(handle, HIP_R_16F, HIP_R_16F, row, ta, tb,
                                batches, zero_beta, device, inplace, row,
                                passed);
            run<hip_bfloat16, hip_bfloat16>(handle, HIP_R_16BF, HIP_R_16BF, row,
                                            ta, tb, batches, zero_beta, device,
                                            inplace, row, passed);
            run<__half, float>(handle, HIP_R_16F, HIP_R_32F, row, ta, tb,
                               batches, zero_beta, device, inplace, row,
                               passed);
            run<hip_bfloat16, float>(handle, HIP_R_16BF, HIP_R_32F, row, ta, tb,
                                     batches, zero_beta, device, inplace, row,
                                     passed);
          }
  run<float, float>(handle, HIP_R_32F, HIP_R_32F, false, false, false, 1, false,
                    false, false, false, passed, 0);
  run<float, float>(handle, HIP_R_32F, HIP_R_32F, false, false, false, 1, false,
                    false, false, false, passed, 23, true);
  run<float, float>(handle, HIP_R_32F, HIP_R_32F, true, false, false, 1, true,
                    true, false, false, passed, 23, true);
  for (auto epilogue : {HIPBLASLT_EPILOGUE_BIAS, HIPBLASLT_EPILOGUE_RELU,
                        HIPBLASLT_EPILOGUE_RELU_BIAS, HIPBLASLT_EPILOGUE_GELU,
                        HIPBLASLT_EPILOGUE_GELU_BIAS})
    for (bool row : {false, true})
      for (bool ta : {false, true})
        for (bool tb : {false, true})
          for (int batches : {1, 3}) {
            const auto bias_type = ta ? HIP_R_32F : HIPBLASLT_DATATYPE_INVALID;
            run<float, float>(handle, HIP_R_32F, HIP_R_32F, row, ta, tb,
                              batches, false, ta != tb, !ta, row, passed, 23,
                              false, epilogue, bias_type);
            run<__half, __half>(handle, HIP_R_16F, HIP_R_16F, row, ta, tb,
                                batches, false, ta != tb, !ta, row, passed, 23,
                                false, epilogue, bias_type);
            run<hip_bfloat16, hip_bfloat16>(
                handle, HIP_R_16BF, HIP_R_16BF, row, ta, tb, batches, false,
                ta != tb, !ta, row, passed, 23, false, epilogue, bias_type);
          }
  lt_check(hipblasLtDestroy(handle));
  std::printf("HIPBLASLT_SCALAR_PASS cases=%d\n", passed);
}
