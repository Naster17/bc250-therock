// SPDX-License-Identifier: MIT
// Fixed-size device chunks reuse N2.8's budget/deflation/failure semantics.
#include "schur_control_n28.hip"

template <class R>
__global__ void small_schur_chunk(const Z<R> *t, const Z<R> *q, Z<R> *out_t,
                                  Z<R> *out_q, int *state, int n, int budget,
                                  int chunk_size) {
  __shared__ Z<R> current_t[256], current_q[256];
  __shared__ Z<R> work[256], r[256], u[256], left[256], v[16], shift;
  __shared__ int m, phase;
  int lane = threadIdx.x;
  if (lane < n)
    for (int j = 0; j < n; ++j) {
      current_t[lane * n + j] = t[lane * n + j];
      current_q[lane * n + j] = q[lane * n + j];
    }
  __syncthreads();
  for (int step = 0; step < chunk_size; ++step) {
    schur_control_core(current_t, current_q, out_t, out_q, state, n, budget,
                       work, r, u, left, v, shift, m, phase);
    // Covers both terminal and running exits from the reused core, including
    // roundoff truncation. All lanes take the same terminal branch.
    __syncthreads();
    if (state[4] != 0)
      break;
    // Separate shared input buffers prevent in-place Q update aliasing. A row
    // needs all its old columns when forming each new column.
    if (step + 1 < chunk_size && lane < n)
      for (int j = 0; j < n; ++j) {
        current_t[lane * n + j] = out_t[lane * n + j];
        current_q[lane * n + j] = out_q[lane * n + j];
      }
    __syncthreads();
  }
}

extern "C" int bc250_schur_chunk(const void *t, const void *q, void *ot,
                                 void *oq, int *state, int n, int budget,
                                 int chunk_size, int fp64, void *stream) {
  if (!t || !q || !ot || !oq || !state || n < 2 || n > 16 || budget < 0 ||
      budget > 4096 || chunk_size < 1 || chunk_size > 32 ||
      (fp64 != 0 && fp64 != 1))
    return int(hipErrorInvalidValue);
#define LAUNCH(R)                                                              \
  hipLaunchKernelGGL(small_schur_chunk<R>, dim3(1), dim3(32), 0,               \
                     reinterpret_cast<hipStream_t>(stream),                    \
                     static_cast<const Z<R> *>(t),                             \
                     static_cast<const Z<R> *>(q), static_cast<Z<R> *>(ot),    \
                     static_cast<Z<R> *>(oq), state, n, budget, chunk_size)
  if (fp64) {
    LAUNCH(double);
  } else {
    LAUNCH(float);
  }
#undef LAUNCH
  return int(hipGetLastError());
}
