// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipfft/hipfft.h>
#include <rocblas/rocblas.h>
#include <rocrand/rocrand.h>
#include <rocsolver/rocsolver.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(operation)                                                       \
  do {                                                                         \
    const int status = static_cast<int>(operation);                            \
    if (status != 0) {                                                         \
      std::fprintf(stderr, "%s: status=%d\n", #operation, status);             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

static void close(float actual, double expected, double tolerance,
                  const char *operation, std::size_t index) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) > tolerance * (1 + std::abs(expected))) {
    std::fprintf(stderr, "%s[%zu]: got=%g expected=%g\n", operation, index,
                 actual, expected);
    std::exit(1);
  }
}

int main() {
  constexpr int m = 17, n = 33, k = 31;
  std::vector<float> a(m * k), b(n * k), c(m * n), result(m * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = (static_cast<int>(i % 13) - 6) / 8.0f;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = (static_cast<int>(i % 7) - 3) / 4.0f;
  }
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = static_cast<int>(i % 11) - 5;
  }
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK(hipMalloc(&da, a.size() * sizeof(float)));
  CHECK(hipMalloc(&db, b.size() * sizeof(float)));
  CHECK(hipMalloc(&dc, c.size() * sizeof(float)));
  CHECK(
      hipMemcpy(da, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice));
  CHECK(
      hipMemcpy(db, b.data(), b.size() * sizeof(float), hipMemcpyHostToDevice));
  CHECK(
      hipMemcpy(dc, c.data(), c.size() * sizeof(float), hipMemcpyHostToDevice));
  const float alpha = 0.5f, beta = 0.125f;
  rocblas_handle handle{};
  CHECK(rocblas_create_handle(&handle));
  CHECK(rocblas_sgemm(handle, rocblas_operation_none,
                      rocblas_operation_transpose, m, n, k, &alpha, da, m, db,
                      n, &beta, dc, m));
  CHECK(hipMemcpy(result.data(), dc, c.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < m; ++row) {
      double product = 0;
      for (int inner = 0; inner < k; ++inner) {
        product +=
            static_cast<double>(a[row + inner * m]) * b[column + inner * n];
      }
      const auto index = row + column * m;
      close(result[index], alpha * product + beta * c[index], 1e-6, "rocblas",
            index);
    }
  }
  std::puts("ROCBLAS_GEMM_PASS");
  CHECK(
      hipMemcpy(dc, c.data(), c.size() * sizeof(float), hipMemcpyHostToDevice));
  hipblasHandle_t wrapper{};
  CHECK(hipblasCreate(&wrapper));
  CHECK(hipblasSgemm(wrapper, HIPBLAS_OP_N, HIPBLAS_OP_T, m, n, k, &alpha, da,
                     m, db, n, &beta, dc, m));
  CHECK(
      hipMemcpy(c.data(), dc, c.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < c.size(); ++i) {
    close(c[i], result[i], 0, "hipblas", i);
  }
  CHECK(hipblasDestroy(wrapper));
  CHECK(hipFree(dc));
  CHECK(hipFree(db));
  CHECK(hipFree(da));
  std::puts("HIPBLAS_GEMM_PASS");

  float matrix[] = {5, 1, 2, 1, 6, 1, 2, 1, 7};
  double reference[9]{};
  for (int column = 0; column < 3; ++column) {
    for (int row = column; row < 3; ++row) {
      double value = matrix[row + column * 3];
      for (int inner = 0; inner < column; ++inner) {
        value -= reference[row + inner * 3] * reference[column + inner * 3];
      }
      reference[row + column * 3] =
          row == column ? std::sqrt(value)
                        : value / reference[column + column * 3];
    }
  }
  float *device_matrix = nullptr;
  int *device_info = nullptr, info = -1;
  CHECK(hipMalloc(&device_matrix, sizeof(matrix)));
  CHECK(hipMalloc(&device_info, sizeof(info)));
  CHECK(
      hipMemcpy(device_matrix, matrix, sizeof(matrix), hipMemcpyHostToDevice));
  CHECK(rocsolver_spotrf(handle, rocblas_fill_lower, 3, device_matrix, 3,
                         device_info));
  CHECK(hipMemcpy(&info, device_info, sizeof(info), hipMemcpyDeviceToHost));
  CHECK(
      hipMemcpy(matrix, device_matrix, sizeof(matrix), hipMemcpyDeviceToHost));
  if (info != 0) {
    std::fprintf(stderr, "potrf info=%d\n", info);
    return 1;
  }
  for (int column = 0; column < 3; ++column) {
    for (int row = column; row < 3; ++row) {
      close(matrix[row + column * 3], reference[row + column * 3], 1e-6,
            "potrf", row + column * 3);
    }
  }
  CHECK(hipFree(device_info));
  CHECK(hipFree(device_matrix));
  CHECK(rocblas_destroy_handle(handle));
  std::puts("ROCSOLVER_CHOLESKY_PASS");

  constexpr int length = 17;
  hipfftComplex input[length], output[length];
  for (int i = 0; i < length; ++i) {
    input[i] = {static_cast<float>(i % 5 - 2), static_cast<float>(i % 3 - 1)};
  }
  hipfftComplex *device_fft = nullptr;
  CHECK(hipMalloc(&device_fft, sizeof(input)));
  CHECK(hipMemcpy(device_fft, input, sizeof(input), hipMemcpyHostToDevice));
  hipfftHandle fft{};
  CHECK(hipfftPlan1d(&fft, length, HIPFFT_C2C, 1));
  CHECK(hipfftExecC2C(fft, device_fft, device_fft, HIPFFT_FORWARD));
  CHECK(hipMemcpy(output, device_fft, sizeof(output), hipMemcpyDeviceToHost));
  for (int frequency = 0; frequency < length; ++frequency) {
    double real = 0, imaginary = 0;
    for (int i = 0; i < length; ++i) {
      const double angle = -2 * std::acos(-1.0) * frequency * i / length;
      real += input[i].x * std::cos(angle) - input[i].y * std::sin(angle);
      imaginary += input[i].x * std::sin(angle) + input[i].y * std::cos(angle);
    }
    close(output[frequency].x, real, 1e-5, "fft real", frequency);
    close(output[frequency].y, imaginary, 1e-5, "fft imaginary", frequency);
  }
  CHECK(hipfftDestroy(fft));
  CHECK(hipFree(device_fft));
  std::puts("HIPFFT_PRIME_DFT_PASS");

  constexpr std::size_t count = 16384;
  std::vector<float> first(count), second(count);
  float *device_random = nullptr;
  CHECK(hipMalloc(&device_random, count * sizeof(float)));
  for (auto *samples : {&first, &second}) {
    rocrand_generator generator{};
    CHECK(
        rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK(rocrand_set_seed(generator, 1234));
    CHECK(rocrand_generate_uniform(generator, device_random, count));
    CHECK(hipMemcpy(samples->data(), device_random, count * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(rocrand_destroy_generator(generator));
  }
  double mean = 0, squares = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(first[i]) || first[i] <= 0 || first[i] > 1 ||
        first[i] != second[i]) {
      std::fprintf(stderr, "Random bounds/reproducibility mismatch at %zu\n",
                   i);
      return 1;
    }
    mean += first[i] / count;
    squares += static_cast<double>(first[i]) * first[i] / count;
  }
  if (std::abs(mean - 0.5) > 0.015 ||
      std::abs(squares - mean * mean - 1.0 / 12) > 0.01) {
    std::fprintf(stderr, "Random mean=%g variance=%g\n", mean,
                 squares - mean * mean);
    return 1;
  }
  CHECK(hipFree(device_random));
  std::puts("ROCRAND_REPRODUCIBILITY_DISTRIBUTION_PASS");
  CHECK(hipDeviceSynchronize());
  std::puts("MATH_SMOKE_PASS");
}
