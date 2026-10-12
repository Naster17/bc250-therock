// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>

static void hip_check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}

static void sparse_check(rocsparse_status status) {
  if (status != rocsparse_status_success)
    throw std::runtime_error("rocSPARSE operation failed");
}

template <typename T> static bool run(int analyses) {
  constexpr int side = 7, rows = side * side;
  std::vector<int> offsets(rows + 1), columns;
  std::vector<T> values;
  for (int row = 0; row < rows; ++row) {
    offsets[row] = columns.size();
    for (int column : {row - side, row - 1, row, row + 1, row + side}) {
      if (column < 0 || column >= rows ||
          (column == row - 1 && row % side == 0) ||
          (column == row + 1 && row % side == side - 1))
        continue;
      columns.push_back(column);
      values.push_back(column == row ? T(4) : T(-1));
    }
  }
  offsets[rows] = columns.size();
  const int nnz = columns.size();
  int *device_offsets, *device_columns;
  T *device_values, *device_x, *device_y;
  hip_check(hipMalloc(&device_offsets, offsets.size() * sizeof(int)));
  hip_check(hipMalloc(&device_columns, columns.size() * sizeof(int)));
  hip_check(hipMalloc(&device_values, values.size() * sizeof(T)));
  hip_check(hipMalloc(&device_x, rows * sizeof(T)));
  hip_check(hipMalloc(&device_y, (rows + 16) * sizeof(T)));
  hip_check(hipMemcpy(device_offsets, offsets.data(),
                      offsets.size() * sizeof(int), hipMemcpyHostToDevice));
  hip_check(hipMemcpy(device_columns, columns.data(),
                      columns.size() * sizeof(int), hipMemcpyHostToDevice));
  hip_check(hipMemcpy(device_values, values.data(), values.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  rocsparse_handle handle;
  rocsparse_mat_descr descriptor;
  rocsparse_mat_info info = nullptr;
  sparse_check(rocsparse_create_handle(&handle));
  sparse_check(rocsparse_create_mat_descr(&descriptor));
  sparse_check(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
  if (analyses) {
    sparse_check(rocsparse_create_mat_info(&info));
    for (int iteration = 0; iteration < analyses; ++iteration) {
      if constexpr (std::is_same_v<T, float>)
        sparse_check(rocsparse_scsrmv_analysis(
            handle, rocsparse_operation_none, rows, rows, nnz, descriptor,
            device_values, device_offsets, device_columns, info));
      else
        sparse_check(rocsparse_dcsrmv_analysis(
            handle, rocsparse_operation_none, rows, rows, nnz, descriptor,
            device_values, device_offsets, device_columns, info));
    }
  }
  bool passed = true;
  for (int repeat = 0; repeat < 10; ++repeat)
    for (bool ones : {true, false})
      for (T beta : {T(0), T(0.5)}) {
        T alpha = repeat % 2 ? T(1) : T(1.5);
        std::vector<T> x(rows), y(rows + 16, T(-17));
        for (int row = 0; row < rows; ++row)
          x[row] = ones ? T(1) : T(row % 7 - 3);
        hip_check(hipMemcpy(device_x, x.data(), rows * sizeof(T),
                            hipMemcpyHostToDevice));
        hip_check(hipMemcpy(device_y, y.data(), y.size() * sizeof(T),
                            hipMemcpyHostToDevice));
        if constexpr (std::is_same_v<T, float>)
          sparse_check(rocsparse_scsrmv(
              handle, rocsparse_operation_none, rows, rows, nnz, &alpha,
              descriptor, device_values, device_offsets, device_columns, info,
              device_x, &beta, device_y));
        else
          sparse_check(rocsparse_dcsrmv(
              handle, rocsparse_operation_none, rows, rows, nnz, &alpha,
              descriptor, device_values, device_offsets, device_columns, info,
              device_x, &beta, device_y));
        hip_check(hipMemcpy(y.data(), device_y, y.size() * sizeof(T),
                            hipMemcpyDeviceToHost));
        int mismatches = 0;
        for (int row = 0; row < rows; ++row) {
          T sum = 0;
          for (int entry = offsets[row]; entry < offsets[row + 1]; ++entry)
            sum += values[entry] * x[columns[entry]];
          T expected = alpha * sum + beta * T(-17);
          if (y[row] != expected) {
            if (mismatches++ == 0)
              fprintf(stderr,
                      "SPMV_MISMATCH type=%s analysis=%d repeat=%d ones=%d "
                      "beta=%g row=%d actual=%g expected=%g\n",
                      std::is_same_v<T, float> ? "float" : "double", analyses,
                      repeat, ones, double(beta), row, double(y[row]),
                      double(expected));
          }
        }
        for (int row = rows; row < rows + 16; ++row)
          if (y[row] != T(-17))
            ++mismatches;
        passed = passed && mismatches == 0;
      }
  if (info)
    sparse_check(rocsparse_destroy_mat_info(info));
  sparse_check(rocsparse_destroy_mat_descr(descriptor));
  sparse_check(rocsparse_destroy_handle(handle));
  hip_check(hipFree(device_y));
  hip_check(hipFree(device_x));
  hip_check(hipFree(device_values));
  hip_check(hipFree(device_columns));
  hip_check(hipFree(device_offsets));
  printf("LEGACY_SPMV type=%s analysis=%d cases=40 pass=%d\n",
         std::is_same_v<T, float> ? "float" : "double", analyses, passed);
  return passed;
}

int main() {
  try {
    bool passed = true;
    for (int analyses : {0, 1, 2}) {
      passed = run<float>(analyses) && passed;
      passed = run<double>(analyses) && passed;
    }
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
