# N2.1 complex Hermitian eigensolver qualification — 2026-10-15

The existing native `torch.linalg.eigh` path passed22 new checks on .104, followed
by retained evidence from the32-case P2 solver regression. No backend defect was
found within this scope, so no Torch/hipSOLVER/rocSOLVER code or selection policy
was changed. Normal `torch.linalg.eigh(A, UPLO='L')` / `UPLO='U'` already selects
this path for GPU complex inputs; no opt-in or CPU fallback was added.

## Existing coverage and selected gap

`build_tools/bc250/validate_priority2_math.py --group solver` has28 checks across
float32/float64/complex64/complex128: batched9x9 positive-definite solve,
Cholesky, eigenvalues, singular values, inverse residual,13x7 QR reconstruction/
orthogonality and solve gradients. Four further real diagonal-system checks
cover known solution/residual at condition10,000. This did not qualify Hermitian
eigenvectors, indefinite spectra, repeated-eigenvalue subspaces, the upper/lower
triangle contract or larger blocked reduction sizes. P2 is not duplicated as a
new implementation; its32 checks were rerun unchanged on the current consumer.

## Pinned consumer and selection audit

Runtime identities and SHA256s: `n21/runtime-identity.json`. Active Torch source
is4ee65717d69e9f51021272e3f37d0f342f68fc08; ROCm libraries source is
a6f7d3d8ba7a1d022628d8c3fdcc54235551a9b8. Kernel remains image84995f1 plus
AMDGPU0957f1, same boot a287fd16-c97b-4d88-8055-b0553a2f7631.

- Torch `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp`,
  `linalg_eigh_cusolver`: ROCm complex inputs use `linalg_eigh_cusolver_syevd`.
  The optional direct rocSOLVER strided-batched branch is restricted to real
  float/double. Complex batches are iterated in `apply_syevd`, not handled by a
  newly implemented complex batched-dispatch optimization.
- `cuda/linalg/CUDASolver.h` enables the generic64 API for ROCm>=7.14;
  this consumer reports HIP7.17/ROCm10.2. `CUDASolver.cpp` complex `xsyevd`
  specializations use the datatype-selected Xsyevd API, hipified to
  `hipsolverDnXsyevd` with C32F/R32F or C64F/R64F pairs.
- hipSOLVER `library/src/amd_detail/hipsolver_dense64.cpp` queries/allocates
  workspace and dispatches to native `rocsolver_cheevd_64`/`zheevd_64` when
  HIPSOLVER_ENABLE_EIGENSOLVERS_64 is enabled; the compatibility branch uses
  checked32-bit dimensions. The loaded DSO imports both64 eigenvector symbols
  (`nm -D`); retained build.ninja defines confirm the enabled branch
  (`n21/dispatch-audit.json`). Legacy DnCheevd/Zheevd wrappers
  also exist, but are not the selected generic64 Torch call.
- rocSOLVER `roclapack_syevd_heevd.cpp/.hpp` implements this path. Source defaults
  in `include/ideal_sizes.hpp` set HETRD block64, unblocked/blocked switch256 and
  two-stage switch11000. Size257 crosses the default blocked threshold; neither
  runtime algorithm tracing nor the11000 two-stage path was qualified here.

`n21/eigh-blocked-command.json` records read-only mounts from the active prefix,
not staging libraries. The test records mapped hipSOLVER/rocSOLVER/rocBLAS DSOs.
Input/output device checks, GPU stream ordering and the audited native dispatch
establish GPU solver execution; CPU complex128 work supplies independent reference
comparisons. There is no claim that reference construction runs on the GPU.

## New bounded checks

`build_tools/bc250/validate_solver_n21.py` tests complex64 and complex128, two-matrix
batches at sizes17/65/129/257 with separated indefinite spectra, and size33 with
three repeated eigenvalue groups (-3,0,2). Both UPLO choices use padded strided
GPU input with NaNs only in the ignored triangle; input and all guard lanes must
remain unchanged. Calls run on a nondefault stream with explicit producer/
consumer waits. These20 cases compare CPU complex128 eigenvalues, reconstruction,
normalized residual, orthogonality and spectral projectors. Projectors avoid
invalid raw-vector phase/basis comparisons within repeated eigenspaces.

Two additional11x11 complex64/complex128 gradient cases compare a phase-invariant
weighted eigenvalue/eigenvector objective against CPU complex128 gradients with
separated eigenvalues. No gradient claim is made at repeated eigenvalues.

| Maximum over final fixtures | complex64 | complex128 |
| --- | ---: | ---: |
| Normalized eigenpair residual | 1.15e-6 | 2.16e-15 |
| Orthogonality error / sqrt(n) | 1.33e-6 | 2.58e-15 |
| Reconstruction relative error | 1.38e-6 | 2.60e-15 |
| Subspace projector error / sqrt(rank) | 3.16e-5 | 7.69e-14 |
| Gradient relative error | 8.10e-7 | 1.97e-15 |

Fixed residual/orthogonality/reconstruction bounds are8e-5/2e-11 for complex64/
complex128; projector bound is four times that. Eigenvalue and gradient checks
also use fixed allclose bounds recorded in the test/results. No tolerances were
relaxed in response to a numerical failure.

## Execution and evidence

Run under the existing exclusive runtime lock using the existing bounded runner:

```sh
ws=/home/nik/bc250work/rocm-10.2-gfx1013
python3 "$ws/helpers/update_dev_prefix.py" -- \
  python3 "$ws/helpers/validate_tiled_sdpa_on_host.py" \
  --label UNIQUE_SOLVER_LABEL --script validate_solver_n21.py
```

Stage the test in WS/helpers first. The runner requires root on .104, checks
watchdog/no other containers, starts a fresh600s recovery guardian, runs the
pinned container with native gfx1013 devices and SDMA disabled, records logs,
then completes its deadline after container cleanup. Test failures remain failures
in result/exit evidence even when cleanup safely completes the recovery deadline.

Final labels `solver-eigh-n21-blocked` (22) and `solver-p2-regression-n21` (32)
exit0, dmesg byte-identical, guards acknowledged. `solver-eigh-n21-r2` earlier
passed18 cases before adding257. The initial `solver-eigh-n21` had16 harness
failures from multiplying batched vectors by eigenvalues without an explicit
row singleton; this reference-check broadcasting bug was corrected. Initial
logs/results are retained and are not a backend defect/fix claim.

All source/JSON/log evidence is under `n21/` (copied terminal logs normalize CRLF
to LF); untouched full logs and kernel logs remain in matching
.104 WS/logs directories. No native code changed, so no native rebuild or new
cache was needed. Existing builds, caches, prefix metadata, checkpoints, boot/
power policy, .105 and user-owned src/ were retained. No agent job remains.

Limits: one Hermitian operation, finite used triangle, bounded dimensions/batches,
first-order eager gradients only on separated spectra. No general condition-number,
extreme-scaling/nonfinite-input, non-Hermitian eig/SVD, arbitrary blocked/two-stage,
concurrency, performance or sustained qualification. Original Python CPU crash and
separate MIOpen elapsed-time/workspace failure remain unresolved; the fixed CSR/
VM-heap corruption is a different defect.

Next N2.2: audit existing nonsymmetric `torch.linalg.eig` coverage and pinned
consumer dispatch; select a bounded complex eigenpair gap with CPU/residual
references and patch only a demonstrated failure.

N2.2 follow-up: [nonsymmetric eig compatibility and native gap](N22.md). The
non-Hermitian operation uses host LAPACK in the pinned hipSOLVER; it must not
inherit the native-GPU classification of Hermitian eigh above.
