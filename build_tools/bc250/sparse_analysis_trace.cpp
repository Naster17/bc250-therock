// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Diagnostic-only preload for the pinned rocSPARSE source. Never ship this
// private-layout inspection as part of the runtime API or deployment.
#include "rocsparse_csrmv_info.hpp"
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <vector>

static void dump(rocsparse_mat_info info, const char *where) {
  if (!info)
    return;
  // The pinned _rocsparse_mat_info has no base/vtable and csrmv_info is its
  // first member. Copy its representation without an aliasing cast.
  rocsparse_csrmv_info matrix = nullptr;
  std::memcpy(&matrix, info, sizeof(matrix));
  if (!matrix)
    return;
  const auto &adaptive = matrix->adaptive;
  fprintf(stderr,
          "CSR_METADATA where=%s m=%lld n=%lld nnz=%lld blocks=%zu first=%lld "
          "last=%lld\n",
          where, (long long)matrix->m, (long long)matrix->n,
          (long long)matrix->nnz, adaptive.size, (long long)adaptive.first_row,
          (long long)adaptive.last_row);
  if (!adaptive.size || adaptive.size > 128 || !adaptive.row_blocks ||
      !adaptive.wg_ids)
    return;
  std::vector<int> rows(adaptive.size), groups(adaptive.size);
  auto row_status = hipMemcpy(rows.data(), adaptive.row_blocks,
                              rows.size() * sizeof(int), hipMemcpyDeviceToHost);
  auto group_status =
      hipMemcpy(groups.data(), adaptive.wg_ids, groups.size() * sizeof(int),
                hipMemcpyDeviceToHost);
  fprintf(stderr, "CSR_METADATA_COPY rows_status=%d groups_status=%d",
          int(row_status), int(group_status));
  for (size_t i = 0; i < rows.size(); ++i)
    fprintf(stderr, " (%d,%d)", rows[i], groups[i]);
  fputc('\n', stderr);
}

#define TRACE_ANALYSIS(NAME, TYPE)                                             \
  extern "C" rocsparse_status NAME(                                            \
      rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m,     \
      rocsparse_int n, rocsparse_int nnz, const rocsparse_mat_descr descr,     \
      const TYPE *values, const rocsparse_int *rows,                           \
      const rocsparse_int *columns, rocsparse_mat_info info) {                 \
    static auto real =                                                         \
        reinterpret_cast<decltype(&NAME)>(dlsym(RTLD_NEXT, #NAME));            \
    if (!real)                                                                 \
      return rocsparse_status_internal_error;                                  \
    auto status =                                                              \
        real(handle, trans, m, n, nnz, descr, values, rows, columns, info);    \
    if (status == rocsparse_status_success)                                    \
      dump(info, #NAME);                                                       \
    return status;                                                             \
  }

#define TRACE_SPMV(NAME, TYPE)                                                 \
  extern "C" rocsparse_status NAME(                                            \
      rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m,     \
      rocsparse_int n, rocsparse_int nnz, const TYPE *alpha,                   \
      const rocsparse_mat_descr descr, const TYPE *values,                     \
      const rocsparse_int *rows, const rocsparse_int *columns,                 \
      rocsparse_mat_info info, const TYPE *x, const TYPE *beta, TYPE *y) {     \
    static auto real =                                                         \
        reinterpret_cast<decltype(&NAME)>(dlsym(RTLD_NEXT, #NAME));            \
    if (!real)                                                                 \
      return rocsparse_status_internal_error;                                  \
    dump(info, #NAME);                                                         \
    return real(handle, trans, m, n, nnz, alpha, descr, values, rows, columns, \
                info, x, beta, y);                                             \
  }

TRACE_ANALYSIS(rocsparse_scsrmv_analysis, float)
TRACE_ANALYSIS(rocsparse_dcsrmv_analysis, double)
TRACE_SPMV(rocsparse_scsrmv, float)
TRACE_SPMV(rocsparse_dcsrmv, double)
