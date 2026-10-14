// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test-only observer: forward every call unchanged and count successful
// matmuls.
#include <atomic>
#include <dlfcn.h>
#include <hipblaslt/hipblaslt.h>

namespace {
std::atomic<unsigned long long> successful_calls{0};
}

extern "C" unsigned long long bc250_hipblaslt_successful_calls() {
  return successful_calls.load(std::memory_order_relaxed);
}

extern "C" hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
                const void *alpha, const void *a, hipblasLtMatrixLayout_t ad,
                const void *b, hipblasLtMatrixLayout_t bd, const void *beta,
                const void *c, hipblasLtMatrixLayout_t cd, void *d,
                hipblasLtMatrixLayout_t dd, const hipblasLtMatmulAlgo_t *algo,
                void *workspace, size_t workspace_size, hipStream_t stream) {
  static const auto real = [] {
    auto *symbol = dlsym(RTLD_NEXT, "hipblasLtMatmul");
    if (!symbol) {
      // Torch can load its dependencies with RTLD_LOCAL. Resolve the real
      // library explicitly instead of making the observer force a fallback.
      static void *library = dlopen("libhipblaslt.so", RTLD_NOW | RTLD_LOCAL);
      if (library)
        symbol = dlsym(library, "hipblasLtMatmul");
    }
    return reinterpret_cast<decltype(&hipblasLtMatmul)>(symbol);
  }();
  if (!real)
    return HIPBLAS_STATUS_NOT_INITIALIZED;
  const auto status = real(handle, desc, alpha, a, ad, b, bd, beta, c, cd, d,
                           dd, algo, workspace, workspace_size, stream);
  if (status == HIPBLAS_STATUS_SUCCESS)
    successful_calls.fetch_add(1, std::memory_order_relaxed);
  return status;
}
