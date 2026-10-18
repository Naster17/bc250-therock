// SPDX-License-Identifier: MIT
// Private cap16 recovery stage. Python must check finite normalized upper T and
// global pairwise diagonal separation >8sqrt(eps), then check output/residual.
#include "schur_qr_n27.hip"

template <class R>
__global__ void small_triangular_backsub(const Z<R> *t, Z<R> *out, int n) {
  int column = threadIdx.x;
  if (column >= n)
    return;
  // Columns are independent. Each lane keeps its own homogeneous eigenvector;
  // no shared state, global handle/workspace or cross-stream output reuse.
  Z<R> y[16];
  for (int k = 0; k < n; ++k)
    y[k] = {R(k == column), 0};
  for (int i = n - 2; i >= 0; --i) {
    if (column > i) {
      Z<R> rhs{0, 0};
      for (int k = i + 1; k < n; ++k)
        rhs = rhs + t[i * n + k] * y[k];
      Z<R> pivot = t[i * n + i] - t[column * n + column];
      // |T|<=1, |y|<=1, n<=16; checked active gaps bound division and
      // squared pivots away from underflow for both qualified precisions.
      Z<R> numerator = rhs * pivot.conj();
      R divisor = pivot.norm2();
      y[i] = {-numerator.x / divisor, -numerator.y / divisor};
    }
    // Match the reference's per-row, per-column homogeneous rescaling.
    R magnitude = 0;
    for (int k = 0; k < n; ++k)
      magnitude = fmax(magnitude, hypot(y[k].x, y[k].y));
    for (int k = 0; k < n; ++k)
      y[k] = {y[k].x / magnitude, y[k].y / magnitude};
  }
  for (int k = 0; k < n; ++k)
    out[k * n + column] = y[k];
}

extern "C" int bc250_triangular_backsub(const void *t, void *out, int n,
                                        int fp64, void *stream) {
  if (!t || !out || t == out || n < 1 || n > 16 || (fp64 != 0 && fp64 != 1))
    return int(hipErrorInvalidValue);
#define LAUNCH(R)                                                              \
  hipLaunchKernelGGL(small_triangular_backsub<R>, dim3(1), dim3(32), 0,        \
                     reinterpret_cast<hipStream_t>(stream),                    \
                     static_cast<const Z<R> *>(t), static_cast<Z<R> *>(out),   \
                     n)
  if (fp64) {
    LAUNCH(double);
  } else {
    LAUNCH(float);
  }
#undef LAUNCH
  return int(hipGetLastError());
}
