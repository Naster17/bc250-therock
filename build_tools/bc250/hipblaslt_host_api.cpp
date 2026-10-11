// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_status(hipblasStatus_t actual, hipblasStatus_t expected,
                         const char *operation) {
  if (actual != expected) {
    std::fprintf(stderr, "%s: status=%d expected=%d\n", operation,
                 static_cast<int>(actual), static_cast<int>(expected));
    std::exit(1);
  }
}

int main() {
  hipblasLtHandle_t handle{};
  hipblasLtMatmulDesc_t description{};
  hipblasLtMatrixLayout_t layout{};
  hipblasLtMatmulPreference_t preference{};
  check_status(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS,
               "create handle");
  check_status(
      hipblasLtMatmulDescCreate(&description, HIPBLAS_COMPUTE_32F, HIP_R_32F),
      HIPBLAS_STATUS_SUCCESS, "create description");
  check_status(hipblasLtMatrixLayoutCreate(&layout, HIP_R_32F, 16, 16, 16),
               HIPBLAS_STATUS_SUCCESS, "create layout");
  check_status(hipblasLtMatmulPreferenceCreate(&preference),
               HIPBLAS_STATUS_SUCCESS, "create preference");

  hipblasLtMatmulHeuristicResult_t result{};
  int count = 99;
  check_status(hipblasLtMatmulAlgoGetHeuristic(handle, description, layout,
                                               layout, layout, layout,
                                               preference, 1, &result, &count),
               HIPBLAS_STATUS_NOT_SUPPORTED, "query unavailable GEMM backend");
  if (count != 0) {
    std::fprintf(stderr, "Host-only heuristic advertised %d algorithms\n",
                 count);
    return 1;
  }

  std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
  check_status(hipblaslt_ext::getAllAlgos(
                   handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                   HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_32F, HIP_R_32F, HIP_R_32F,
                   HIP_R_32F, HIPBLAS_COMPUTE_32F, algorithms),
               HIPBLAS_STATUS_NOT_SUPPORTED,
               "enumerate unavailable algorithms");
  if (!algorithms.empty()) {
    std::fprintf(stderr, "Host-only extension advertised algorithms\n");
    return 1;
  }

  void *data = nullptr;
  if (hipMalloc(&data, 16 * 16 * sizeof(float)) != hipSuccess) {
    return 1;
  }
  const float alpha = 1, beta = 0;
  check_status(hipblasLtMatmul(handle, description, &alpha, data, layout, data,
                               layout, &beta, data, layout, data, layout,
                               nullptr, nullptr, 0, nullptr),
               HIPBLAS_STATUS_NOT_SUPPORTED,
               "execute unavailable GEMM backend");
  if (hipFree(data) != hipSuccess) {
    return 1;
  }

  check_status(hipblasLtMatmulPreferenceDestroy(preference),
               HIPBLAS_STATUS_SUCCESS, "destroy preference");
  check_status(hipblasLtMatrixLayoutDestroy(layout), HIPBLAS_STATUS_SUCCESS,
               "destroy layout");
  check_status(hipblasLtMatmulDescDestroy(description), HIPBLAS_STATUS_SUCCESS,
               "destroy description");
  check_status(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS,
               "destroy handle");
  std::puts("HIPBLASLT_HOST_API_PASS");
}
