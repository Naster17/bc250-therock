# B02.1: explicit bounded CSR solve fallback

This batch adds the opt-in `bc250_sparse` module. It converts a **small** canonical
CSR system to dense storage on the input GPU, solves it with the existing
rocSOLVER-backed Torch dense LU path, and checks LU info, finite results and a
normalized residual. It does not change `torch.sparse.spsolve`, which still rejects
this ROCm build. There is no CPU linear solver or global dispatch replacement.

Final candidate and installed runtime each pass **381 checks**:253 new helper,
80 retained triangular edge cases,41 sparse and seven core checks.

Qualification and installed identities are recorded in [summary.json](summary.json).
The canonical [flow](../../../flow.md) identifies the current accepted revision;
failed candidates remain evidence, not deployed capabilities.

## Use

The module is installed in the development image's Python environment, available
through `bc250-torch` and `bc250-train`. Import and call it explicitly:

```python
import bc250_sparse

# A: square GPU CSR tensor; b: same-dtype GPU vector or matrix of RHS columns.
x = bc250_sparse.spsolve(A, b)
```

For stored-value gradients, including complex values, use the CSR parts API:

```python
# crow/col: matching GPU int32 or int64 indices, canonical zero-based CSR.
# values: GPU FP32/FP64/complex64/complex128, may require grad.
# b: same device/dtype, may require grad.
x = bc250_sparse.solve_csr(crow, col, values, b)
loss = x.abs().square().mean()
loss.backward()
```

Both APIs solve `AX=B` and return a dense GPU tensor of the input dtype. The CSR
wrapper supports real stored-value gradients and real/complex RHS gradients.
Complex **CSR tensor** autograd is refused explicitly: this Torch revision rejects
constructing a complex compressed sparse tensor from differentiable values.
The parts API avoids that restriction. A separate complex CSR values-view failure
(`Sparse CSR tensors do not have strides`) is avoided by materializing detached
values for the wrapper; complex matrix-value gradients must use the parts API.

The first adjoint solve and matrix product run with AMP disabled, as does the
forward solve. This avoids BF16 rounding of matrix-value gradients when backward
is called inside BF16 autocast. Sparsity is fixed; derivatives are with respect to
stored values, including explicitly stored zeros, and RHS values. Indices have no
derivatives. Higher derivatives, compile and graph capture are unsupported.

## Admission and memory boundary

- Unbatched square systems, `1 <= n <= 1024`; vector or `1..64` RHS columns.
- Matching FP32, FP64, complex64 or complex128 data on the same GPU. BF16/FP16
  are refused; there is no implicit promotion or device transfer.
- Canonical CSR: valid monotonic row pointers, sorted unique columns within each
  row, no out-of-range indices, at most `n*n` scalar stored values.
- Before densification, `element_size*(8*n*n + 6*n*nrhs)` must fit the configured
  budget. `max_dense_bytes` defaults to 64MiB and can only lower that ceiling.
  This estimates major dense intermediates; it is **not** a cap on total process,
  allocator, autograd or native workspace memory. The budget can reject a size
  below the dimension ceiling, especially with complex128 data.
- Exact singular LU info, nonfinite inputs/results/residual scale, or normalized
  residual above `64*eps(input real dtype)` raise. The residual uses FP64/C128
  arithmetic and per-RHS infinity norms. It does not establish good conditioning,
  uniqueness under a numerical rank threshold, or a small forward error.

Storage remains O(n²), factorization work O(n³), and backward currently performs
an adjoint dense solve. Validation and status checks synchronize GPU scalar
results to the host. No speed, reduced memory, large-sparse scalability, factor
reuse, right solve, `out`, batching or universal stability claim is made.

## Why this path

The pinned Torch `_apply_sparse_csr_linear_solve` explicitly rejects `USE_ROCM`.
Existing N2.13 CSR/BSR triangular solves do not perform a general factorization.
In the pinned hipSOLVER AMD implementation, `csrlsvqr` itself allocates a dense
`n*n` matrix and calls rocSOLVER `gels`; calling that API would not supply a
scalable sparse solution. Its sparse Cholesky route uses host CHOLMOD symbolic
analysis and is specific to the positive-definite case. rocSOLVER `csrrf_analysis`
requires previously computed factors/permutations and matching sparsity; it is
not a general initial sparse factorizer. These sources were inspected before
implementation; native general sparse factorization remains an open B02 scope.

## Qualification and failures retained

The same fixed gates compare CPU double/complex-double outputs and all stored
value/RHS leaf gradients, plus independent adjoint formulas and directional
finite differences. They cover four dtypes, int32/int64 indices, vector/multiple
RHS, strided offset views and unused-storage zeros, private streams, BF16 AMP
contexts, explicit stored zeros, pivoting, larger analytic diagonal systems,
admission/budget/singular/nonfinite/residual failures, and a real GPU trace.
Retained N2.13 exact-pivot edge cases, sparse operations and core checks accompany
the new helper. Exact accepted counts/results are in the summary.

Failures were retained and corrected without relaxing numerical tolerances:

1. Entry Podman terminal launches failed before Python because `/var/tmp` was
   missing. Restoring that standard directory with mode1777 restored a terminal
   `/bin/true` launch and the GPU audit. No package or OCI runtime was replaced.
   Upstream conmon's [console socket code](https://github.com/containers/conmon/blob/v2.1.13/src/conn_sock.c)
   creates a temporary socket path; the local logs contain the observed failure.
2. The initial audit exposed complex compressed sparse constructor autograd
   rejection. The revised parts audit passes real/complex values and RHS gradients.
3. Candidate-r1 failed a strict FP32 value-gradient comparison under BF16 AMP;
   the explicit first adjoint path fixes precision without changing the gate.
4. Candidate-r2 exposed the complex CSR values-view error; the wrapper now copies
   those non-differentiable values, while the parts API retains value gradients.
5. Candidate-r3 reached the invalid-column refusal test, where native checked
   CSR construction triggered a GPU device assertion and a process abort/cleanup
   segfault. The final helper does **not** call that GPU invariant checker: safe
   predicates reject bad row/column metadata before repeat/scatter consumption.
   A fresh-process seven-check core recovery gate passed without a reboot.

The device-assert cleanup crash is a known trigger in this test. It does not
resolve or identify every historical Python CPU segfault. The older fixed VM-heap/
CSR corruption and B04.2 queue repair remain separate. Native malformed-CSR
handling outside this helper is unchanged.

## Reproduce and recovery

On `.104`, `WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Run host controllers as
root under `WS/helpers/update_dev_prefix.py -- ...`, using the existing exclusive
runtime lock. The staged candidate is recorded in `logs/sparse-b021-stage-r4`;
qualification uses `qualify_sparse_b021_on_host.py --attempt r4`. GPU gates use
the existing acknowledged 600-second guard and explicit r2/r3 kernel selection.
This pure Python image layer requires no native build or ccache operation.

`deploy_sparse_b021_on_host.py` installs only after the candidate gate and runs
an installed gate, restoring exact prior metadata/image automatically on failure.
For an explicit rollback, under the exclusive lock run:

```text
python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/update_dev_prefix.py -- python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/deploy_sparse_b021_on_host.py --rollback
```

The exact prior runtime is `WS/recovery/sparse-b021-before/runtime.json` and its
shared qualifier is retained there. Old images, sources, builds, caches, releases
and fallback links remain. The user application is excluded from inspection and
tests (`include_training_app=False`); `.105`, BIOS, kernel and boot fallbacks are
untouched. Owned raw logs, module variants, guards and recovery metadata are
exported under local `kernel/artifacts/sparse-b021/`.
