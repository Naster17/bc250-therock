// SPDX-License-Identifier: MIT
// Isolated order<=16 experiment, not a production rocSOLVER entry point.
#include <cmath>
#include <hip/hip_runtime.h>

template <class R> struct Z {
  R x, y;
  __device__ Z operator+(Z b) const { return {x + b.x, y + b.y}; }
  __device__ Z operator-(Z b) const { return {x - b.x, y - b.y}; }
  __device__ Z operator*(Z b) const {
    return {x * b.x - y * b.y, x * b.y + y * b.x};
  }
  __device__ Z operator*(R b) const { return {x * b, y * b}; }
  __device__ Z conj() const { return {x, -y}; }
  __device__ R norm2() const { return x * x + y * y; }
};

template <class R>
__device__ void qr_similarity_core(const Z<R> *t, const Z<R> *q,
                                   const Z<R> *shift, Z<R> *out_t, Z<R> *out_q,
                                   int n, int m, Z<R> *r, Z<R> *u, Z<R> *left,
                                   Z<R> *v) {
  // One wave cooperates on shared matrices. No host reads inside the QR step.
  // Inputs are component-normalized by the bounded Schur driver.
  int lane = threadIdx.x;
  if (lane < m)
    for (int j = 0; j < m; ++j) {
      int i = lane;
      r[i * m + j] = t[i * n + j] - (i == j ? *shift : Z<R>{0, 0});
      u[i * m + j] = {R(i == j), 0};
    }
  __syncthreads();
  for (int k = 0; k < m; ++k) {
    if (lane == 0) {
      R norm = 0;
      for (int i = k; i < m; ++i) {
        v[i] = r[i * m + k];
        norm += v[i].norm2();
      }
      norm = sqrt(norm);
      if (norm != 0) {
        R first = sqrt(v[k].norm2());
        Z<R> phase = first == 0 ? Z<R>{1, 0} : v[k] * (R(1) / first);
        v[k] = v[k] + phase * norm;
        R vnorm = 0;
        for (int i = k; i < m; ++i)
          vnorm += v[i].norm2();
        vnorm = sqrt(vnorm);
        for (int i = k; i < m; ++i)
          v[i] = v[i] * (R(1) / vnorm);
      }
      // Exact zero column leaves v=0, so H=I without divergent barriers.
    }
    __syncthreads();
    // R <- H R and U <- U H, H=I-2vv^H.
    if (lane >= k && lane < m) {
      int j = lane;
      Z<R> dot{0, 0};
      for (int i = k; i < m; ++i)
        dot = dot + v[i].conj() * r[i * m + j];
      for (int i = k; i < m; ++i)
        r[i * m + j] = r[i * m + j] - (v[i] * dot) * R(2);
    }
    if (lane < m) {
      int i = lane;
      Z<R> dot{0, 0};
      for (int j = k; j < m; ++j)
        dot = dot + u[i * m + j] * v[j];
      for (int j = k; j < m; ++j)
        u[i * m + j] = u[i * m + j] - (dot * v[j].conj()) * R(2);
    }
    __syncthreads();
  }
  // Full similarity, including coupling to the deflated trailing block.
  if (lane < n) {
    int i = lane;
    for (int j = 0; j < n; ++j) {
      Z<R> sum{0, 0};
      if (i < m)
        for (int k = 0; k < m; ++k)
          sum = sum + u[k * m + i].conj() * t[k * n + j];
      else
        sum = t[i * n + j];
      left[i * n + j] = sum;
    }
  }
  __syncthreads();
  if (lane < n) {
    int i = lane;
    for (int j = 0; j < n; ++j) {
      Z<R> st{0, 0}, sq{0, 0};
      if (j < m) {
        for (int k = 0; k < m; ++k) {
          st = st + left[i * n + k] * u[k * m + j];
          sq = sq + q[i * n + k] * u[k * m + j];
        }
      } else {
        st = left[i * n + j];
        sq = q[i * n + j];
      }
      out_t[i * n + j] = st;
      out_q[i * n + j] = sq;
    }
  }
}

template <class R>
__global__ void small_qr_similarity(const Z<R> *t, const Z<R> *q,
                                    const Z<R> *shift, Z<R> *out_t, Z<R> *out_q,
                                    int n, int m) {
  __shared__ Z<R> r[256], u[256], left[256], v[16];
  qr_similarity_core(t, q, shift, out_t, out_q, n, m, r, u, left, v);
}

extern "C" int bc250_qr_step(const void *t, const void *q, const void *shift,
                             void *ot, void *oq, int n, int m, int fp64,
                             void *stream) {
  if (!t || !q || !shift || !ot || !oq || n < 2 || n > 16 || m < 2 || m > n ||
      (fp64 != 0 && fp64 != 1))
    return int(hipErrorInvalidValue);
#define LAUNCH(R)                                                              \
  hipLaunchKernelGGL(                                                          \
      small_qr_similarity<R>, dim3(1), dim3(32), 0,                            \
      reinterpret_cast<hipStream_t>(stream), static_cast<const Z<R> *>(t),     \
      static_cast<const Z<R> *>(q), static_cast<const Z<R> *>(shift),          \
      static_cast<Z<R> *>(ot), static_cast<Z<R> *>(oq), n, m)
  if (fp64) {
    LAUNCH(double);
  } else {
    LAUNCH(float);
  }
#undef LAUNCH
  return int(hipGetLastError());
}
