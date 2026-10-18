// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#define main bc250FloatScalarMain
#include "hipblaslt_scalar_smoke.cpp"
#undef main
#include <cstdint>
#include <cstring>
#include <string>

static void setop(hipblasLtMatmulDesc_t op, hipblasLtMatmulDescAttributes_t key,
                  const void *data, size_t bytes) {
  lt_check(hipblasLtMatmulDescSetAttribute(op, key, data, bytes));
}
static int32_t signed_bits(uint32_t bits) {
  int32_t out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}
static void execute(hipblasLtHandle_t handle, int m, int n, int k, int mode,
                    int count, bool extreme = false) {
  const bool ta = mode & 1, tb = mode & 2, device = count % 2;
  const bool inplace = count % 5 == 0;
  const int batches = extreme ? 1 : 3;
  const int64_t offsets[] = {5, 7, 9, inplace ? 9 : 11};
  auto make = [&](int rows, int cols, bool row, bool broadcast) {
    const int ld = (row ? cols : rows) + 3;
    return Layout{rows,    cols,
                  ld,      broadcast ? 0 : (row ? rows : cols) * ld + 13,
                  batches, row};
  };
  const auto a = make(ta ? k : m, ta ? m : k, count & 2, count & 1);
  const auto b = make(tb ? n : k, tb ? k : n, count & 4, count & 2);
  const auto c = make(m, n, count & 8, !inplace);
  const auto d = inplace ? c : make(m, n, count & 8, false);
  std::vector<int8_t> ha(std::max(size_t(1), a.size() + offsets[0]), -51),
      hb(std::max(size_t(1), b.size() + offsets[1]), -51);
  std::vector<int32_t> hc(c.size() + offsets[2], 0x5a5a5a5a),
      hd(d.size() + offsets[3], 0x5a5a5a5a);
  for (int z = 0; z < batches; ++z) {
    for (int i = 0; i < a.rows; ++i)
      for (int j = 0; j < a.cols; ++j)
        ha[offsets[0] + a.index(i, j, z)] =
            extreme ? (k == 1041 ? 127 : -128)
                    : int8_t((i * 7 + j * 3 + z * 11) % 256 - 128);
    for (int i = 0; i < b.rows; ++i)
      for (int j = 0; j < b.cols; ++j)
        hb[offsets[1] + b.index(i, j, z)] =
            extreme ? (k == 1041 || k > 200000 ? 127 : -128)
                    : int8_t((i * 3 + j * 5 + z * 17) % 256 - 128);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        hc[offsets[2] + c.index(i, j, z)] =
            (i + j) % 2 ? INT32_MAX - i : INT32_MIN + j;
  }
  if (inplace)
    hd = hc;
  const int32_t as[] = {-3, 0, 1, INT32_MIN};
  const int32_t bs[] = {0, 2, -1, INT32_MAX};
  int32_t alpha = extreme ? 1 : as[(count / 2) % 4],
          beta = extreme ? 0 : bs[(count / 3) % 4];
  const hipblasLtEpilogue_t epis[] = {
      HIPBLASLT_EPILOGUE_DEFAULT, HIPBLASLT_EPILOGUE_BIAS,
      HIPBLASLT_EPILOGUE_RELU, HIPBLASLT_EPILOGUE_RELU_BIAS};
  const auto epi = extreme ? HIPBLASLT_EPILOGUE_DEFAULT : epis[(count / 4) % 4];
  const bool has_bias =
      epi == HIPBLASLT_EPILOGUE_BIAS || epi == HIPBLASLT_EPILOGUE_RELU_BIAS;
  const int64_t bias_stride = count % 3 ? m + 4 : 0;
  std::vector<int32_t> bias(size_t(batches - 1) * bias_stride + m + 1, 0);
  for (size_t i = 0; i < bias.size(); ++i)
    bias[i] = i % 2 ? INT32_MAX - int32_t(i) : -int32_t(i) - 17;
  auto expected = hd;
  for (int z = 0; z < batches; ++z)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        int64_t sum = 0;
        for (int p = 0; p < k; ++p)
          sum += int64_t(ha[offsets[0] + a.index(ta ? p : i, ta ? i : p, z)]) *
                 int64_t(hb[offsets[1] + b.index(tb ? j : p, tb ? p : j, z)]);
        uint32_t bits =
            uint32_t(alpha) * uint32_t(sum) +
            uint32_t(beta) * uint32_t(hc[offsets[2] + c.index(i, j, z)]);
        if (has_bias)
          bits += uint32_t(bias[size_t(z) * bias_stride + i]);
        int32_t value = signed_bits(bits);
        if (epi == HIPBLASLT_EPILOGUE_RELU ||
            epi == HIPBLASLT_EPILOGUE_RELU_BIAS)
          value = std::max(int32_t(0), value);
        expected[offsets[3] + d.index(i, j, z)] = value;
      }
  auto A = upload(ha), B = upload(hb);
  auto C = upload(hc), D = inplace ? C : upload(hd), Bias = upload(bias);
  auto av = upload(std::vector<int32_t>{alpha}),
       bv = upload(std::vector<int32_t>{beta});
  auto ad = descriptor(HIP_R_8I, a), bd = descriptor(HIP_R_8I, b),
       cd = descriptor(HIP_R_32I, c), dd = descriptor(HIP_R_32I, d);
  hipblasLtMatrixLayout_t layouts[] = {ad, bd, cd, dd};
  for (int i = 0; i < 4; ++i)
    lt_check(hipblasLtMatrixLayoutSetAttribute(layouts[i],
                                               HIPBLASLT_MATRIX_LAYOUT_OFFSET,
                                               &offsets[i], sizeof(int64_t)));
  hipblasLtMatmulDesc_t op{};
  lt_check(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));
  const auto opa = ta ? HIPBLAS_OP_T : HIPBLAS_OP_N,
             opb = tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const auto pm =
      device ? HIPBLASLT_POINTER_MODE_DEVICE : HIPBLASLT_POINTER_MODE_HOST;
  setop(op, HIPBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa));
  setop(op, HIPBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb));
  setop(op, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));
  setop(op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
  if (has_bias) {
    const auto type = HIP_R_32I;
    setop(op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &Bias, sizeof(Bias));
    setop(op, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &type, sizeof(type));
    setop(op, HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &bias_stride,
          sizeof(bias_stride));
  }
  hipblasLtMatmulPreference_t pref{};
  lt_check(hipblasLtMatmulPreferenceCreate(&pref));
  hipblasLtMatmulHeuristicResult_t result{};
  int returned = 0;
  lt_check(hipblasLtMatmulAlgoGetHeuristic(handle, op, ad, bd, cd, dd, pref, 1,
                                           &result, &returned));
  if (returned != 1 || result.workspaceSize != 0 ||
      result.state != HIPBLAS_STATUS_SUCCESS)
    std::exit(1);
  hipStream_t stream{};
  hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  const bool null_ab = !device && (alpha == 0 || k == 0);
  const bool null_c = !device && beta == 0 && !inplace;
  lt_check(hipblasLtMatmul(
      handle, op, device ? av : &alpha, null_ab ? nullptr : A, ad,
      null_ab ? nullptr : B, bd, device ? bv : &beta, null_c ? nullptr : C, cd,
      D, dd, count % 2 ? &result.algo : nullptr, nullptr, 0, stream));
  hip_check(hipStreamSynchronize(stream));
  hip_check(hipMemcpy(hd.data(), D, hd.size() * 4, hipMemcpyDeviceToHost));
  if (hd != expected) {
    for (size_t i = 0; i < hd.size(); ++i)
      if (hd[i] != expected[i]) {
        std::fprintf(stderr, "INT8 mismatch case=%d at%zu got%d expected%d\n",
                     count, i, hd[i], expected[i]);
        break;
      }
    std::exit(1);
  }
  hip_check(hipStreamDestroy(stream));
  lt_check(hipblasLtMatmulPreferenceDestroy(pref));
  lt_check(hipblasLtMatmulDescDestroy(op));
  for (auto x : layouts)
    lt_check(hipblasLtMatrixLayoutDestroy(x));
  hip_check(hipFree(A));
  hip_check(hipFree(B));
  hip_check(hipFree(C));
  if (!inplace)
    hip_check(hipFree(D));
  hip_check(hipFree(Bias));
  hip_check(hipFree(av));
  hip_check(hipFree(bv));
  std::printf("PASS direct-int8 case=%d m=%d n=%d k=%d\n", count, m, n, k);
}

static int refusals(hipblasLtHandle_t handle) {
  Layout x{4, 4, 4, 16, 1, false};
  auto a = descriptor(HIP_R_8I, x), b = descriptor(HIP_R_8I, x),
       c = descriptor(HIP_R_32I, x);
  hipblasLtMatmulDesc_t op{};
  lt_check(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));
  hipblasLtMatmulPreference_t pref{};
  lt_check(hipblasLtMatmulPreferenceCreate(&pref));
  int count = 0;
  auto query = [&](hipblasStatus_t expected) {
    int n = 7;
    hipblasLtMatmulHeuristicResult_t r{};
    lt_check(hipblasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c, pref, 1,
                                             &r, &n),
             expected);
    if (n != 0)
      std::exit(1);
    ++count;
  };
  auto epi = HIPBLASLT_EPILOGUE_GELU;
  setop(op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
  query(HIPBLAS_STATUS_NOT_SUPPORTED);
  epi = HIPBLASLT_EPILOGUE_BIAS;
  setop(op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
  query(HIPBLAS_STATUS_INVALID_VALUE);
  auto buf = upload(std::vector<int32_t>(64, 1));
  void *ptr = buf;
  setop(op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &ptr, sizeof(ptr));
  auto type = HIP_R_32F;
  setop(op, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &type, sizeof(type));
  query(HIPBLAS_STATUS_NOT_SUPPORTED);
  epi = HIPBLASLT_EPILOGUE_DEFAULT;
  setop(op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
  ptr = nullptr;
  setop(op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &ptr, sizeof(ptr));
  auto pm = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
  setop(op, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));
  query(HIPBLAS_STATUS_NOT_SUPPORTED);
  pm = HIPBLASLT_POINTER_MODE_HOST;
  setop(op, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));
  int32_t one = 1, zero = 0;
  auto call = [&](const void *alpha, const void *A, const void *B,
                  const void *beta, const void *C, void *D) {
    lt_check(hipblasLtMatmul(handle, op, alpha, A, a, B, b, beta, C, c, D, c,
                             nullptr, nullptr, 0, nullptr),
             HIPBLAS_STATUS_INVALID_VALUE);
    ++count;
  };
  call(nullptr, buf, buf, &zero, nullptr, buf + 32);
  call(&one, nullptr, buf, &zero, nullptr, buf + 32);
  call(&one, buf, buf, &one, nullptr, buf + 32);
  call(&one, buf, buf, &zero, nullptr, reinterpret_cast<char *>(buf + 32) + 1);
  call(&one, buf, buf, &zero, nullptr, reinterpret_cast<char *>(buf) + 4);
  call(&one, buf, buf, &one, buf + 33, buf + 32);
  pm = HIPBLASLT_POINTER_MODE_DEVICE;
  setop(op, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));
  call(buf + 32, buf, buf, buf + 20, buf, buf + 32);
  lt_check(hipblasLtMatmulDescDestroy(op));
  lt_check(hipblasLtMatmulPreferenceDestroy(pref));
  lt_check(hipblasLtMatrixLayoutDestroy(a));
  lt_check(hipblasLtMatrixLayoutDestroy(b));
  lt_check(hipblasLtMatrixLayoutDestroy(c));
  hip_check(hipFree(buf));
  return count;
}
int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  hipDeviceProp_t props{};
  hip_check(hipGetDeviceProperties(&props, 0));
  if (std::string(props.gcnArchName).substr(0, 7) != "gfx1013")
    return 1;
  hipblasLtHandle_t handle{};
  lt_check(hipblasLtCreate(&handle));
  int count = 0;
  for (auto shape : std::vector<std::vector<int>>{{3, 5, 0},
                                                  {1, 7, 1},
                                                  {17, 19, 23},
                                                  {32, 24, 32},
                                                  {65, 33, 65},
                                                  {2, 2, 1041}})
    for (int orders = 0; orders < 4; ++orders)
      for (int trans = 0; trans < 4; ++trans)
        execute(handle, shape[0], shape[1], shape[2], trans, count++);
  execute(handle, 1, 1, 131073, 0, count++, true);
  execute(handle, 1, 1, 262145, 0, count++, true);
  execute(handle, 3, 5, 1041, 0, count++, true);
  const int rejected = refusals(handle);
  lt_check(hipblasLtDestroy(handle));
  std::printf("DENSE_INT8_A08_PASS cases=%d refusals=%d\n", count, rejected);
}
