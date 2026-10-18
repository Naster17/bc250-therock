// SPDX-License-Identifier: MIT
// Isolated bounded iteration. The N2.7 shared-memory QR arithmetic is reused.
#include "schur_qr_n27.hip"
#include <limits>

template <class R> __device__ R magnitude(Z<R> z) { return hypot(z.x, z.y); }

template <class R> __device__ Z<R> principal_sqrt(Z<R> z) {
  R w = sqrt((magnitude(z) + fabs(z.x)) * R(0.5));
  if (w == 0)
    return {0, z.y};
  if (z.x >= 0)
    return {w, z.y / (R(2) * w)};
  return {fabs(z.y) / (R(2) * w), copysign(w, z.y)};
}

// State: active, iterations, deflations, since_deflation, phase.
// phase=0 continue,1 converged,2 budget exhausted,3 nonfinite/excess fill.
// Each launch first consumes trailing deflations, then performs at most one QR.
template <class R>
__device__ void schur_control_core(const Z<R> *t, const Z<R> *q, Z<R> *out_t,
                                   Z<R> *out_q, int *state, int n, int budget,
                                   Z<R> *work, Z<R> *r, Z<R> *u, Z<R> *left,
                                   Z<R> *v, Z<R> &shift, int &m, int &phase) {
  constexpr R eps = std::numeric_limits<R>::epsilon();
  constexpr R tiny = std::numeric_limits<R>::min();
  int lane = threadIdx.x;
  if (lane < n)
    for (int j = 0; j < n; ++j)
      work[lane * n + j] = t[lane * n + j];
  __syncthreads();
  if (lane == 0) {
    m = state[0];
    phase = 0;
    while (m > 1) {
      R threshold = eps * (magnitude(work[(m - 2) * n + m - 2]) +
                           magnitude(work[(m - 1) * n + m - 1])) +
                    tiny;
      if (!(magnitude(work[(m - 1) * n + m - 2]) <= threshold))
        break;
      work[(m - 1) * n + m - 2] = {0, 0};
      --m;
      ++state[2];
      state[3] = 0;
    }
    state[0] = m;
    if (m == 1)
      phase = 1;
    else if (state[1] == budget)
      phase = 2;
    if (!phase) {
      Z<R> aa = work[(m - 2) * n + m - 2];
      Z<R> bb = work[(m - 2) * n + m - 1];
      Z<R> cc = work[(m - 1) * n + m - 2];
      Z<R> dd = work[(m - 1) * n + m - 1];
      Z<R> middle = (aa + dd) * R(0.5);
      Z<R> half = (aa - dd) * R(0.5);
      Z<R> root = principal_sqrt(half * half + bb * cc);
      Z<R> s1 = middle + root, s2 = middle - root;
      shift = magnitude(s1 - dd) <= magnitude(s2 - dd) ? s1 : s2;
      if (state[3] && state[3] % 20 == 0)
        shift = dd + Z<R>{magnitude(cc), 0};
    }
    state[4] = phase;
  }
  __syncthreads();
  // Phase is uniform for the block; no lane exits across a required barrier.
  if (phase) {
    if (lane < n)
      for (int j = 0; j < n; ++j) {
        out_t[lane * n + j] = work[lane * n + j];
        out_q[lane * n + j] = q[lane * n + j];
      }
    return;
  }
  qr_similarity_core(work, q, &shift, out_t, out_q, n, m, r, u, left, v);
  __syncthreads();
  if (lane == 0) {
    R fill = 0, norm = 0;
    bool finite = true;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        Z<R> z = out_t[i * n + j];
        finite = finite && isfinite(z.x) && isfinite(z.y);
        norm += z.norm2();
        if (i > j + 1)
          fill += z.norm2();
      }
    if (!finite || !(sqrt(fill) <= R(128) * eps * sqrt(norm))) {
      state[4] = 3;
    } else {
      ++state[1];
      ++state[3];
    }
  }
  __syncthreads();
  if (state[4] == 0 && lane < n)
    for (int j = 0; j < lane - 1; ++j)
      out_t[lane * n + j] = {0, 0};
}

template <class R>
__global__ void small_schur_control(const Z<R> *t, const Z<R> *q, Z<R> *out_t,
                                    Z<R> *out_q, int *state, int n,
                                    int budget) {
  __shared__ Z<R> work[256], r[256], u[256], left[256], v[16], shift;
  __shared__ int m, phase;
  schur_control_core(t, q, out_t, out_q, state, n, budget, work, r, u, left, v,
                     shift, m, phase);
}

extern "C" int bc250_schur_control(const void *t, const void *q, void *ot,
                                   void *oq, int *state, int n, int budget,
                                   int fp64, void *stream) {
  if (!t || !q || !ot || !oq || !state || n < 2 || n > 16 || budget < 0 ||
      budget > 4096 || (fp64 != 0 && fp64 != 1))
    return int(hipErrorInvalidValue);
#define LAUNCH(R)                                                              \
  hipLaunchKernelGGL(small_schur_control<R>, dim3(1), dim3(32), 0,             \
                     reinterpret_cast<hipStream_t>(stream),                    \
                     static_cast<const Z<R> *>(t),                             \
                     static_cast<const Z<R> *>(q), static_cast<Z<R> *>(ot),    \
                     static_cast<Z<R> *>(oq), state, n, budget)
  if (fp64) {
    LAUNCH(double);
  } else {
    LAUNCH(float);
  }
#undef LAUNCH
  return int(hipGetLastError());
}
