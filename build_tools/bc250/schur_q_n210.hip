// SPDX-License-Identifier: MIT
// Small GEHRD reflector-Q formation; native GEHRD itself is unchanged.
#include "schur_qr_n27.hip"

template <class R>
__global__ void small_gehrd_form_q(const Z<R> *packed, const Z<R> *tau,
                                   Z<R> *out, int n) {
  // One row per lane. Padding avoids power-of-two row strides in shared Q.
  __shared__ Z<R> q[272], v[16];
  int row = threadIdx.x;
  if (row < n)
    for (int j = 0; j < n; ++j)
      q[row * (n + 1) + j] = {R(row == j), 0};
  __syncthreads();
  for (int i = 0; i < n - 1; ++i) {
    if (row < n)
      v[row] = row <= i       ? Z<R>{0, 0}
               : row == i + 1 ? Z<R>{1, 0}
                              : packed[row + i * n]; // GEHRD column-major.
    __syncthreads();
    if (row < n) {
      Z<R> dot{0, 0};
      for (int k = 0; k < n; ++k)
        dot = dot + q[row * (n + 1) + k] * v[k];
      Z<R> factor = tau[i] * dot;
      for (int j = 0; j < n; ++j)
        q[row * (n + 1) + j] = q[row * (n + 1) + j] - factor * v[j].conj();
    }
    __syncthreads();
  }
  if (row < n)
    for (int j = 0; j < n; ++j)
      out[row * n + j] = q[row * (n + 1) + j];
}

extern "C" int bc250_gehrd_form_q(const void *packed, const void *tau,
                                  void *out, int n, int fp64, void *stream) {
  if (!packed || !tau || !out || n < 1 || n > 16 || (fp64 != 0 && fp64 != 1))
    return int(hipErrorInvalidValue);
#define LAUNCH(R)                                                              \
  hipLaunchKernelGGL(small_gehrd_form_q<R>, dim3(1), dim3(32), 0,              \
                     reinterpret_cast<hipStream_t>(stream),                    \
                     static_cast<const Z<R> *>(packed),                        \
                     static_cast<const Z<R> *>(tau), static_cast<Z<R> *>(out), \
                     n)
  if (fp64) {
    LAUNCH(double);
  } else {
    LAUNCH(float);
  }
#undef LAUNCH
  return int(hipGetLastError());
}
