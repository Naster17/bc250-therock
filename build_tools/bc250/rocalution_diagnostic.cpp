// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rocalution/rocalution.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T>
bool run_case(bool accelerator, bool jacobi, bool inspect_inputs) {
  using namespace rocalution;
  disable_accelerator_rocalution(!accelerator);
  init_rocalution();
  bool passed;
  {
    constexpr int side = 7, rows = side * side, nnz = rows * 5 - side * 4;
    int *offsets = new int[rows + 1], *columns = new int[nnz];
    T *values = new T[nnz];
    int entry = 0;
    for (int row = 0; row < rows; ++row) {
      offsets[row] = entry;
      for (int column : {row - side, row - 1, row, row + 1, row + side}) {
        if (column < 0 || column >= rows)
          continue;
        if ((column == row - 1 && row % side == 0) ||
            (column == row + 1 && row % side == side - 1))
          continue;
        columns[entry] = column;
        values[entry++] = column == row ? T(4) : T(-1);
      }
    }
    offsets[rows] = entry;
    if (entry != nnz)
      throw std::runtime_error("invalid diagnostic CSR matrix");
    LocalMatrix<T> matrix;
    matrix.SetDataPtrCSR(&offsets, &columns, &values, "laplacian", nnz, rows,
                         rows);
    LocalVector<T> x, b, ones;
    if (accelerator) {
      matrix.MoveToAccelerator();
      x.MoveToAccelerator();
      b.MoveToAccelerator();
      ones.MoveToAccelerator();
    }
    x.Allocate("x", rows);
    b.Allocate("b", rows);
    ones.Allocate("ones", rows);
    ones.Ones();
    if (inspect_inputs) {
      std::vector<T> host_ones(rows);
      ones.CopyToHostData(host_ones.data());
      for (int row = 0; row < rows; ++row)
        if (host_ones[row] != T(1)) {
          fprintf(stderr, "ONES_MISMATCH row=%d actual=%.17g\n", row,
                  double(host_ones[row]));
          throw std::runtime_error("rocALUTION vector initialization failed");
        }
      if (!matrix.Check())
        throw std::runtime_error("rocALUTION CSR matrix validation failed");
      printf("CG_INPUTS_PASS type=%s gpu=%d\n",
             std::is_same_v<T, float> ? "float" : "double", accelerator);
    }
    matrix.Apply(ones, &b);
    std::vector<T> host(rows);
    b.CopyToHostData(host.data());
    for (int row = 0; row < rows; ++row) {
      int neighbors = (row >= side) + (row < rows - side) + (row % side > 0) +
                      (row % side < side - 1);
      if (host[row] != T(4 - neighbors)) {
        fprintf(stderr, "SPMV_MISMATCH row=%d actual=%.17g\n", row,
                double(host[row]));
        throw std::runtime_error("diagnostic SpMV differs from CPU");
      }
    }
    x.SetRandomUniform(12345, T(-4), T(6));
    x.CopyToHostData(host.data());
    double initial_square_sum = 0;
    for (T value : host)
      initial_square_sum += double(value) * double(value);
    double initial_norm = x.Norm();
    CG<LocalMatrix<T>, LocalVector<T>, T> solver;
    Jacobi<LocalMatrix<T>, LocalVector<T>, T> preconditioner;
    solver.SetOperator(matrix);
    if (jacobi)
      solver.SetPreconditioner(preconditioner);
    solver.Verbose(0);
    solver.Init(1e-8, 0.0, 1e8, 10000);
    solver.Build();
    solver.Solve(b, &x);
    x.ScaleAdd(T(-1), ones);
    double device_error = x.Norm();
    x.CopyToHostData(host.data());
    double error_square_sum = 0, maximum_error = 0;
    for (T value : host) {
      error_square_sum += double(value) * double(value);
      maximum_error = std::max(maximum_error, std::abs(double(value)));
    }
    const double tolerance = std::is_same_v<T, float> ? 1e-3 : 1e-6;
    passed = std::isfinite(device_error) && device_error < tolerance &&
             std::sqrt(error_square_sum) < tolerance;
    printf(
        "CG_DIAGNOSTIC type=%s gpu=%d jacobi=%d iterations=%d residual=%.17g "
        "initial_norm=%.17g initial_cpu_norm=%.17g device_error=%.17g "
        "cpu_error=%.17g max_error=%.17g pass=%d\n",
        std::is_same_v<T, float> ? "float" : "double", accelerator, jacobi,
        solver.GetIterationCount(), solver.GetCurrentResidual(), initial_norm,
        std::sqrt(initial_square_sum), device_error,
        std::sqrt(error_square_sum), maximum_error, passed);
    solver.Clear();
  }
  stop_rocalution();
  return passed;
}

int main(int argc, char **argv) {
  if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--inspect-inputs"))) {
    fprintf(stderr, "usage: rocalution_diagnostic [--inspect-inputs]\n");
    return 2;
  }
  // Optional readbacks introduce synchronization. The ordinary qualification
  // gate keeps them disabled so they cannot conceal an ordering failure.
  bool inspect_inputs = argc == 2;
  bool passed = true;
  try {
    for (bool accelerator : {false, true})
      for (int repeat = 0; repeat < 5; ++repeat)
        for (bool jacobi : {false, true}) {
          passed =
              run_case<float>(accelerator, jacobi, inspect_inputs) && passed;
          passed =
              run_case<double>(accelerator, jacobi, inspect_inputs) && passed;
        }
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    rocalution::stop_rocalution();
    return 1;
  }
  return passed ? 0 : 1;
}
