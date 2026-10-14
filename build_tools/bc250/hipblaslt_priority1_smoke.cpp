// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <limits>
#include <memory>
#include <vector>

static void hip_check(hipError_t s) {
  if (s != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(s));
    std::exit(1);
  }
}
static void lt_check(hipblasStatus_t s,
                     hipblasStatus_t expected = HIPBLAS_STATUS_SUCCESS) {
  if (s != expected) {
    std::fprintf(stderr, "Lt status=%d expected=%d\n", int(s), int(expected));
    std::exit(1);
  }
}
static void require(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}
struct Layout {
  int rows, cols, ld, stride, batches;
  bool row;
  size_t index(int r, int c, int b) const {
    return size_t(b) * stride + (row ? r * ld + c : c * ld + r);
  }
  size_t size() const { return size_t(batches) * stride + 17; }
};
static hipblasLtMatrixLayout_t descriptor(hipDataType type, Layout x) {
  hipblasLtMatrixLayout_t d{};
  lt_check(hipblasLtMatrixLayoutCreate(&d, type, x.rows, x.cols, x.ld));
  const hipblasLtOrder_t order =
      x.row ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
  const int64_t stride = x.stride;
  lt_check(hipblasLtMatrixLayoutSetAttribute(d, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                             &order, sizeof(order)));
  lt_check(hipblasLtMatrixLayoutSetAttribute(
      d, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &x.batches, sizeof(x.batches)));
  lt_check(hipblasLtMatrixLayoutSetAttribute(
      d, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride)));
  return d;
}
template <class T> static T *upload(const std::vector<T> &x) {
  T *d = nullptr;
  hip_check(hipMalloc(&d, x.size() * sizeof(T)));
  hip_check(
      hipMemcpy(d, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}
static bool bias_mode(hipblasLtEpilogue_t e) {
  return e == HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT ||
         e == HIPBLASLT_EPILOGUE_GELU_AUX_BIAS ||
         e == HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT ||
         e == HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT;
}
static bool aux_mode(hipblasLtEpilogue_t e) {
  return e == HIPBLASLT_EPILOGUE_GELU_AUX ||
         e == HIPBLASLT_EPILOGUE_GELU_AUX_BIAS ||
         e == HIPBLASLT_EPILOGUE_DGELU ||
         e == HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT ||
         e == HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT;
}
static bool clamp_mode(hipblasLtEpilogue_t e) {
  return e == HIPBLASLT_EPILOGUE_CLAMP_EXT ||
         e == HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT ||
         e == HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT ||
         e == HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT;
}
static bool swish_mode(hipblasLtEpilogue_t e) {
  return e == HIPBLASLT_EPILOGUE_SWISH_EXT ||
         e == HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT;
}
static double gelu(double x) {
  return 0.5 * x *
         (1 + std::tanh(std::sqrt(2 / std::acos(-1.0)) *
                        (x + 0.044715 * x * x * x)));
}
// Central finite differences are independent of the device's analytic
// derivative.
static double dgelu(double x) {
  constexpr double h = 1e-5;
  return (gelu(x + h) - gelu(x - h)) / (2 * h);
}

template <class T> struct Problem {
  int m, n, k, batches;
  bool ta, tb;
  Layout a, b, c, d, aux;
  hipDataType type;
  hipblasComputeType_t compute;
  hipblasLtEpilogue_t epilogue;
  std::vector<T> A, B, C, D;
  std::vector<float> Bias, Aux;
  T *da = nullptr, *db = nullptr, *dc = nullptr, *dd = nullptr;
  float *bias = nullptr, *e = nullptr;
  hipblasLtMatmulDesc_t op{};
  hipblasLtMatrixLayout_t ad{}, bd{}, cd{}, ddesc{};
  float alpha = 0.7f, beta = -0.3f, act0 = 0, act1 = 0;
  Problem(hipDataType t, int shape, bool trans_a, bool trans_b, bool row,
          hipblasLtEpilogue_t epi,
          float arg0 = std::numeric_limits<float>::quiet_NaN(),
          float arg1 = std::numeric_limits<float>::quiet_NaN())
      : m(shape < 0 ? 129 : 7 + shape), n(shape < 0 ? 131 : 11 - shape),
        k(shape < 0 ? 33 : 13 + shape), batches(2), ta(trans_a), tb(trans_b),
        type(t), epilogue(epi) {
    act0 = std::isnan(arg0) ? (clamp_mode(epi)   ? -0.2f
                               : swish_mode(epi) ? 1.0f
                                                 : 0.0f)
                            : arg0;
    act1 = std::isnan(arg1) ? (clamp_mode(epi) ? 0.3f : 0.0f) : arg1;
    compute = type == HIP_R_16F    ? HIPBLAS_COMPUTE_32F_FAST_16F
              : type == HIP_R_16BF ? HIPBLAS_COMPUTE_32F_FAST_16BF
                                   : HIPBLAS_COMPUTE_32F;
    auto make = [=](int r, int c, bool rm) {
      int ld = (rm ? c : r) + 3;
      return Layout{r, c, ld, (rm ? r : c) * ld + 9, batches, rm};
    };
    a = make(ta ? k : m, ta ? m : k, row);
    b = make(tb ? n : k, tb ? k : n, row);
    c = make(m, n, row);
    d = make(m, n, row);
    aux = make(m, n, false);
    A.assign(a.size(), T(-77));
    B.assign(b.size(), T(-77));
    C.assign(c.size(), T(-77));
    D.assign(d.size(), T(-77));
    Bias.resize(m);
    Aux.assign(aux.size(), -77);
    for (int batch = 0; batch < batches; ++batch) {
      for (int r = 0; r < a.rows; ++r)
        for (int col = 0; col < a.cols; ++col)
          A[a.index(r, col, batch)] =
              T(float((r * 7 + col * 3 + batch * 5) % 17 - 8) / 13);
      for (int r = 0; r < b.rows; ++r)
        for (int col = 0; col < b.cols; ++col)
          B[b.index(r, col, batch)] =
              T(float((r * 2 + col * 11 + batch * 3) % 19 - 9) / 17);
      for (int r = 0; r < m; ++r)
        for (int col = 0; col < n; ++col) {
          C[c.index(r, col, batch)] =
              T(float((r + col * 3 + batch) % 11 - 5) / 7);
          if (epi == HIPBLASLT_EPILOGUE_DGELU)
            Aux[aux.index(r, col, batch)] =
                float((r * 2 + col + batch) % 17 - 8) / 3;
        }
    }
    for (int r = 0; r < m; ++r)
      Bias[r] = float((r * 3) % 13 - 6) / 8;
    da = upload(A);
    db = upload(B);
    dc = upload(C);
    dd = upload(D);
    bias = upload(Bias);
    e = upload(Aux);
    lt_check(hipblasLtMatmulDescCreate(&op, compute, HIP_R_32F));
    const hipblasOperation_t opa = ta ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                             opb = tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &opa, sizeof(opa)));
    lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &opb, sizeof(opb)));
    lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                             &epilogue, sizeof(epilogue)));
    if (swish_mode(epi) || clamp_mode(epi)) {
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT, &act0,
          sizeof(act0)));
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT, &act1,
          sizeof(act1)));
    }
    if (bias_mode(epi)) {
      const hipDataType f32 = HIP_R_32F;
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &f32, sizeof(f32)));
    }
    if (aux_mode(epi)) {
      const int64_t ld = aux.ld, stride = aux.stride;
      const hipDataType f32 = HIP_R_32F;
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &e, sizeof(e)));
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof(ld)));
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, &stride,
          sizeof(stride)));
      lt_check(hipblasLtMatmulDescSetAttribute(
          op, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &f32, sizeof(f32)));
    }
    ad = descriptor(type, a);
    bd = descriptor(type, b);
    cd = descriptor(type, c);
    ddesc = descriptor(type, d);
  }
  Problem(const Problem &) = delete;
  ~Problem() {
    clear_descriptors();
    for (void *p : std::initializer_list<void *>{da, db, dc, dd, bias, e})
      hip_check(hipFree(p));
  }
  void clear_descriptors() {
    if (op)
      lt_check(hipblasLtMatmulDescDestroy(op));
    op = nullptr;
    for (auto p : {ad, bd, cd, ddesc})
      if (p)
        lt_check(hipblasLtMatrixLayoutDestroy(p));
    ad = bd = cd = ddesc = nullptr;
  }
  hipblaslt_ext::GemmProblemType problem_type() {
    hipblaslt_ext::GemmProblemType t(ta ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                     tb ? HIPBLAS_OP_T : HIPBLAS_OP_N, type,
                                     type, type, type, compute);
    t.setOrderA(a.row ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL);
    t.setOrderB(b.row ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL);
    return t;
  }
  hipblaslt_ext::GemmInputs inputs() {
    hipblaslt_ext::GemmInputs i;
    i.setA(da);
    i.setB(db);
    i.setC(dc);
    i.setD(dd);
    i.setAlpha(&alpha);
    i.setBeta(&beta);
    if (bias_mode(epilogue))
      i.setBias(bias);
    if (aux_mode(epilogue))
      i.setAux(e);
    return i;
  }
  hipblaslt_ext::GemmEpilogue epilogue_config() {
    hipblaslt_ext::GemmEpilogue ep;
    ep.setMode(epilogue);
    ep.setAct0(act0);
    ep.setAct1(act1);
    ep.setBiasDataType(HIP_R_32F);
    ep.setAuxDataType(HIP_R_32F);
    ep.setAuxLeadingDimension(aux.ld);
    ep.setAuxBatchStride(aux.stride);
    return ep;
  }
  void verify() {
    std::vector<T> got(D.size());
    std::vector<float> gotaux(Aux.size());
    hip_check(hipMemcpy(got.data(), dd, got.size() * sizeof(T),
                        hipMemcpyDeviceToHost));
    hip_check(hipMemcpy(gotaux.data(), e, gotaux.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    std::vector<bool> used(got.size(), false), auxused(gotaux.size(), false);
    const double tol = type == HIP_R_32F   ? 3e-5
                       : type == HIP_R_16F ? 0.002
                                           : 0.016;
    for (int batch = 0; batch < batches; ++batch)
      for (int r = 0; r < m; ++r)
        for (int col = 0; col < n; ++col) {
          double dot = 0;
          for (int j = 0; j < k; ++j)
            dot += double(float(A[a.index(ta ? j : r, ta ? r : j, batch)])) *
                   float(B[b.index(tb ? col : j, tb ? j : col, batch)]);
          double v = alpha * dot + beta * float(C[c.index(r, col, batch)]);
          if (bias_mode(epilogue))
            v += Bias[r];
          const double pre = v;
          if (clamp_mode(epilogue))
            v = std::max(double(act0), std::min(v, double(act1)));
          else if (epilogue == HIPBLASLT_EPILOGUE_SIGMOID)
            v = 1 / (1 + std::exp(-v));
          else if (epilogue == HIPBLASLT_EPILOGUE_SWISH_EXT ||
                   epilogue == HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT)
            v = v / (1 + std::exp(-v));
          else if (epilogue == HIPBLASLT_EPILOGUE_GELU_AUX ||
                   epilogue == HIPBLASLT_EPILOGUE_GELU_AUX_BIAS)
            v = gelu(v);
          else if (epilogue == HIPBLASLT_EPILOGUE_DGELU)
            v *= dgelu(Aux[aux.index(r, col, batch)]);
          const auto index = d.index(r, col, batch);
          used[index] = true;
          if (std::abs(float(got[index]) - v) > tol * (1 + std::abs(v))) {
            std::fprintf(stderr,
                         "mismatch dtype=%d epilogue=%d row=%d ta=%d tb=%d "
                         "index=%zu got=%g expected=%g\n",
                         int(type), int(epilogue), int(d.row), int(ta), int(tb),
                         index, double(float(got[index])), v);
            std::exit(1);
          }
          if (aux_mode(epilogue) && epilogue != HIPBLASLT_EPILOGUE_DGELU) {
            const auto idx = aux.index(r, col, batch);
            auxused[idx] = true;
            require(std::abs(gotaux[idx] - pre) < 3e-5 * (1 + std::abs(pre)),
                    "Aux preactivation mismatch");
          }
        }
    for (size_t j = 0; j < used.size(); ++j)
      if (!used[j])
        require(float(got[j]) == -77, "D padding overwrite");
    for (size_t j = 0; j < auxused.size(); ++j)
      if (!auxused[j])
        require(gotaux[j] == Aux[j], "Aux input or padding overwrite");
  }
};
static void execute(hipblaslt_ext::GemmInstance &g, hipStream_t stream) {
  std::vector<hipblasLtMatmulHeuristicResult_t> results;
  hipblaslt_ext::GemmPreference preference;
  lt_check(g.algoGetHeuristic(7, preference, results));
  require(results.size() == 1, "expected one native algorithm");
  size_t bytes = 999;
  lt_check(g.isAlgoSupported(results[0].algo, bytes));
  require(bytes == 0, "workspace must be zero");
  lt_check(g.run(stream), HIPBLAS_STATUS_NOT_INITIALIZED);
  lt_check(g.initialize(results[0].algo, nullptr, false, stream));
  hipEvent_t start{}, stop{};
  hip_check(hipEventCreate(&start));
  hip_check(hipEventCreate(&stop));
  lt_check(g.run(stream, start, stop));
  hip_check(hipEventSynchronize(stop));
  require(!g.getKernelName().empty() && !g.getSolutionName().empty(),
          "empty native names");
  hip_check(hipEventDestroy(start));
  hip_check(hipEventDestroy(stop));
}
template <class T>
static void
single(hipblasLtHandle_t handle, hipStream_t stream, hipDataType type, bool row,
       bool ta, bool tb, hipblasLtEpilogue_t epi, bool typed, int &passed,
       float arg0 = std::numeric_limits<float>::quiet_NaN(),
       float arg1 = std::numeric_limits<float>::quiet_NaN(), int shape = 0) {
  Problem<T> p(type, shape, ta, tb, row, epi, arg0, arg1);
  auto t = p.problem_type();
  hipblaslt_ext::Gemm g(handle, t.getOpA(), t.getOpB(), type, type, type, type,
                        p.compute);
  if (typed) {
    // Typed API has column-major C/D but permits either A/B order.
    require(!row, "typed C/D must be column major in this probe");
    auto i = p.inputs();
    auto e = p.epilogue_config();
    lt_check(g.setProblem(p.m, p.n, p.k, p.batches, p.a.ld, p.b.ld, p.c.ld,
                          p.d.ld, p.a.stride, p.b.stride, p.c.stride,
                          p.d.stride, e, i, t));
  } else {
    lt_check(g.setProblem(p.op, &p.alpha, p.da, p.ad, p.db, p.bd, &p.beta, p.dc,
                          p.cd, p.dd, p.ddesc));
    std::vector<hipblasLtMatmulHeuristicResult_t> algos;
    lt_check(hipblaslt_ext::getAllAlgos(
        handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, t.getOpA(), t.getOpB(),
        type, type, type, type, p.compute, algos));
    require(algos.size() == 1, "getAllAlgos native result missing");
    size_t bytes = 99;
    lt_check(hipblaslt_ext::matmulIsAlgoSupported(handle, p.op, &p.alpha, p.ad,
                                                  p.bd, &p.beta, p.cd, p.ddesc,
                                                  algos[0].algo, bytes));
    require(bytes == 0, "C descriptor support workspace mismatch");
    const int index = hipblaslt_ext::getIndexFromAlgo(algos[0].algo);
    std::vector<int> indices{index, index + 1};
    std::vector<hipblasLtMatmulHeuristicResult_t> resolved;
    lt_check(hipblaslt_ext::getAlgosFromIndex(handle, indices, resolved));
    require(resolved.size() == 1, "index resolution mismatch");
    require(!hipblaslt_ext::getSolutionNameFromAlgo(handle, resolved[0].algo)
                 .empty(),
            "query name missing");
  }
  const auto reported_type = g.getProblemTypes();
  require(reported_type.getOpA() == t.getOpA() &&
              reported_type.getOpB() == t.getOpB() &&
              reported_type.getOrderA() == t.getOrderA() &&
              reported_type.getOrderB() == t.getOrderB(),
          "cloned problem metadata mismatch");
  p.clear_descriptors();
  const float saved_alpha = p.alpha, saved_beta = p.beta;
  p.alpha = 99;
  p.beta = 99; // The C++ instance must own the values captured by setProblem.
  execute(g, stream);
  p.alpha = saved_alpha;
  p.beta = saved_beta;
  p.verify();
  ++passed;
}
template <class T>
static void grouped(hipblasLtHandle_t handle, hipStream_t stream,
                    hipDataType type, bool typed, int &passed) {
  std::vector<std::unique_ptr<Problem<T>>> p;
  for (int j = 0; j < 3; ++j)
    p.emplace_back(
        std::make_unique<Problem<T>>(type, j, j & 1, j == 2, false,
                                     j == 0 ? HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT
                                     : j == 1 ? HIPBLASLT_EPILOGUE_GELU_AUX
                                              : HIPBLASLT_EPILOGUE_DGELU));
  hipblaslt_ext::GroupedGemm g(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, type, type,
                               type, type, p[0]->compute);
  if (typed) {
    // Full typed GroupedGemm uses one problem type; exercise all heterogeneous
    // shapes.
    for (auto &x : p) {
      x->ta = false;
      x->tb = false;
    }
    // Descriptors/data were generated for different transposes; use descriptor
    // API below for heterogeneous transposes, and fresh N/N inputs for the
    // typed API.
    p.clear();
    for (int j = 0; j < 3; ++j)
      p.emplace_back(std::make_unique<Problem<T>>(
          type, j, false, false, false,
          j == 0   ? HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT
          : j == 1 ? HIPBLASLT_EPILOGUE_GELU_AUX
                   : HIPBLASLT_EPILOGUE_DGELU));
    std::vector<int64_t> m, n, k, b, lda, ldb, ldc, ldd, sa, sb, sc, sd;
    std::vector<hipblaslt_ext::GemmEpilogue> eps;
    std::vector<hipblaslt_ext::GemmInputs> inputs;
    for (auto &x : p) {
      m.push_back(x->m);
      n.push_back(x->n);
      k.push_back(x->k);
      b.push_back(x->batches);
      lda.push_back(x->a.ld);
      ldb.push_back(x->b.ld);
      ldc.push_back(x->c.ld);
      ldd.push_back(x->d.ld);
      sa.push_back(x->a.stride);
      sb.push_back(x->b.stride);
      sc.push_back(x->c.stride);
      sd.push_back(x->d.stride);
      eps.push_back(x->epilogue_config());
      inputs.push_back(x->inputs());
    }
    auto t = p[0]->problem_type();
    lt_check(g.setProblem(m, n, k, b, lda, ldb, ldc, ldd, sa, sb, sc, sd, eps,
                          inputs, t));
    auto bad = m;
    bad.pop_back();
    lt_check(g.setProblem(bad, n, k, b, lda, ldb, ldc, ldd, sa, sb, sc, sd, eps,
                          inputs, t),
             HIPBLAS_STATUS_INVALID_VALUE);
    require(g.getGemmCount() == 0, "failed setProblem retained old work");
    lt_check(g.setProblem(m, n, k, b, lda, ldb, ldc, ldd, sa, sb, sc, sd, eps,
                          inputs, t));
  } else {
    std::vector<hipblasLtMatmulDesc_t> ops;
    std::vector<hipblasLtMatrixLayout_t> a, b, c, d;
    std::vector<void *> alpha, A, B, beta, C, D;
    for (auto &x : p) {
      ops.push_back(x->op);
      a.push_back(x->ad);
      b.push_back(x->bd);
      c.push_back(x->cd);
      d.push_back(x->ddesc);
      alpha.push_back(&x->alpha);
      A.push_back(x->da);
      B.push_back(x->db);
      beta.push_back(&x->beta);
      C.push_back(x->dc);
      D.push_back(x->dd);
    }
    lt_check(g.setProblem(ops, alpha, A, a, B, b, beta, C, c, D, d));
    // The shorthand cannot reinterpret a prior heterogeneous descriptor group.
    // Four problems versus three types used to index past the type vector.
    std::vector<int64_t> shape(4, 1);
    std::vector<hipblaslt_ext::GemmEpilogue> invalid_eps(4);
    std::vector<hipblaslt_ext::GemmInputs> invalid_inputs(4);
    lt_check(
        g.setProblem(shape, shape, shape, shape, invalid_eps, invalid_inputs),
        HIPBLAS_STATUS_INVALID_VALUE);
    require(g.getGemmCount() == 0, "invalid shorthand retained the old group");
    lt_check(g.setProblem(ops, alpha, A, a, B, b, beta, C, c, D, d));
  }
  require(g.getGemmCount() == 3, "group count mismatch");
  if (!typed) {
    auto types = g.getProblemTypes();
    require(types.size() == 3, "group type count mismatch");
    for (size_t j = 0; j < p.size(); ++j) {
      auto expected = p[j]->problem_type();
      require(types[j].getOpA() == expected.getOpA() &&
                  types[j].getOpB() == expected.getOpB(),
              "heterogeneous group transpose metadata mismatch");
    }
  }
  for (auto &x : p)
    x->clear_descriptors();
  execute(g, stream);
  for (auto &x : p) {
    x->verify();
    ++passed;
  }
}
static void rejections(hipblasLtHandle_t handle, hipStream_t stream,
                       int &passed) {
  Problem<float> p(HIP_R_32F, 0, false, false, false,
                   HIPBLASLT_EPILOGUE_DEFAULT);
  hipblaslt_ext::Gemm g(handle, p.op, &p.alpha, p.da, p.ad, p.db, p.bd, &p.beta,
                        p.dc, p.cd, p.dd, p.ddesc);
  std::vector<hipblasLtMatmulHeuristicResult_t> r;
  hipblaslt_ext::GemmPreference pref;
  lt_check(g.algoGetHeuristic(0, pref, r), HIPBLAS_STATUS_INVALID_VALUE);
  ++passed;
  pref.setStreamKTileSchedulingMode(HIPBLASLT_STREAMK_TILE_SCHEDULING_ON);
  lt_check(g.algoGetHeuristic(1, pref, r), HIPBLAS_STATUS_NOT_SUPPORTED);
  ++passed;
  pref.setStreamKTileSchedulingMode(HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF);
  lt_check(g.algoGetHeuristic(1, pref, r));
  auto bad = r[0].algo;
  bad.data[0] ^= 1;
  size_t bytes = 99;
  lt_check(g.isAlgoSupported(bad, bytes), HIPBLAS_STATUS_INVALID_VALUE);
  ++passed;
  lt_check(g.initialize(r[0].algo, nullptr, true, stream),
           HIPBLAS_STATUS_NOT_SUPPORTED);
  ++passed;
  hipblaslt_ext::GemmTuning tuning;
  tuning.setSplitK(2);
  lt_check(g.isAlgoSupported(r[0].algo, tuning, bytes),
           HIPBLAS_STATUS_NOT_SUPPORTED);
  ++passed;
  lt_check(g.initialize(r[0].algo, tuning, nullptr, false, stream),
           HIPBLAS_STATUS_NOT_SUPPORTED);
  ++passed;
  lt_check(g.initialize(r[0].algo, nullptr, false, stream));
  lt_check(g.initialize(bad, nullptr, false, stream),
           HIPBLAS_STATUS_INVALID_VALUE);
  lt_check(g.run(stream), HIPBLAS_STATUS_NOT_INITIALIZED);
  ++passed;
  hipblaslt_ext::GroupedGemm gg(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_32F,
                                HIP_R_32F, HIP_R_32F, HIP_R_32F,
                                HIPBLAS_COMPUTE_32F);
  lt_check(gg.getDefaultValueForDeviceUserArguments(nullptr),
           HIPBLAS_STATUS_NOT_SUPPORTED);
  ++passed;
  lt_check(gg.run(static_cast<void *>(nullptr), stream),
           HIPBLAS_STATUS_NOT_SUPPORTED);
  ++passed;
}
template <class T>
static void sweep(hipblasLtHandle_t h, hipStream_t s, hipDataType type,
                  int &passed) {
  for (auto epi :
       {HIPBLASLT_EPILOGUE_SIGMOID, HIPBLASLT_EPILOGUE_SWISH_EXT,
        HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT, HIPBLASLT_EPILOGUE_GELU_AUX,
        HIPBLASLT_EPILOGUE_GELU_AUX_BIAS, HIPBLASLT_EPILOGUE_DGELU,
        HIPBLASLT_EPILOGUE_CLAMP_EXT, HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT,
        HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT,
        HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT})
    for (int trans = 0; trans < 4; ++trans) {
      single<T>(h, s, type, false, trans & 1, trans & 2, epi, false, passed);
      single<T>(h, s, type, true, trans & 1, trans & 2, epi, false, passed);
      single<T>(h, s, type, false, trans & 1, trans & 2, epi, true, passed);
    }
  // Public SWISH_EXT is fixed beta=1, even when unused ACT_ARG0 is supplied.
  for (float swish_beta : {0.0f, 0.5f, -1.0f, 1.75f})
    for (auto epi :
         {HIPBLASLT_EPILOGUE_SWISH_EXT, HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT}) {
      single<T>(h, s, type, false, false, true, epi, false, passed, swish_beta);
      single<T>(h, s, type, true, true, false, epi, false, passed, swish_beta);
      single<T>(h, s, type, false, true, true, epi, true, passed, swish_beta);
    }
  single<T>(h, s, type, false, false, false, HIPBLASLT_EPILOGUE_CLAMP_EXT, true,
            passed, 0.0f, 0.0f);
  single<T>(h, s, type, true, true, false,
            HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT, false, passed, 0.4f, -0.1f);
  // Register-tiled dispatch: odd dimensions, two batches, padded layouts,
  // every transpose and output type, including auxiliary/activation epilogues.
  for (auto epi :
       {HIPBLASLT_EPILOGUE_DEFAULT, HIPBLASLT_EPILOGUE_GELU_AUX_BIAS,
        HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT, HIPBLASLT_EPILOGUE_DGELU})
    for (int trans = 0; trans < 4; ++trans)
      for (bool row : {false, true})
        single<T>(h, s, type, row, trans & 1, trans & 2, epi, false, passed,
                  std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::quiet_NaN(), -1);
  grouped<T>(h, s, type, false, passed);
  grouped<T>(h, s, type, true, passed);
}
int main() {
  hipDeviceProp_t prop{};
  hip_check(hipGetDeviceProperties(&prop, 0));
  require(std::strncmp(prop.gcnArchName, "gfx1013", 7) == 0,
          "requires native gfx1013");
  hipblasLtHandle_t handle{};
  lt_check(hipblasLtCreate(&handle));
  hipStream_t stream{};
  hip_check(hipStreamCreate(&stream));
  int passed = 0;
  sweep<float>(handle, stream, HIP_R_32F, passed);
  sweep<__half>(handle, stream, HIP_R_16F, passed);
  sweep<hip_bfloat16>(handle, stream, HIP_R_16BF, passed);
  rejections(handle, stream, passed);
  hip_check(hipStreamDestroy(stream));
  lt_check(hipblasLtDestroy(handle));
  std::printf("HIPBLASLT_PRIORITY1_PASS cases=%d\n", passed);
}
