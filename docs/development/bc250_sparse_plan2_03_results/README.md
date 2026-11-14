# PLAN2-03 — native sparse factorization + nonsymmetric-eig disposition

Completed 2026-11-14 on the .104 release (r2 kernel `7.1.5-bc250rocm-r2`,
release `10.2-gfx1013-20261111-138a66f52`, Torch `2.13.0a0+gfx1013`, HIP 7.17.0,
`gfx1013:xnack-`). Third item of [PLAN2.md](../../../PLAN2.md). **Result:
PASSED** — native ILU(0)/IC(0) sparse factorization with a qualified
preconditioned iterative solve and RHS first gradient, essentially at machine
precision; nonsymmetric eig retains the qualified host-LAPACK default (documented
decision, no rework).

## Headline

**42/42 GPU qualification checks.** Native general-CSR sparse factorization
(hipSPARSE `csrilu02`/`csric02`) + triangular solves (`csrsv2`) + an
ILU-preconditioned Richardson solve, all opt-in with no native operator,
dispatch, default or source change. This is the scalable native alternative to
the B02.1 dense-LU fallback (which densifies and does not scale).

| Metric | FP32 | FP64 | Bound |
| --- | ---: | ---: | --- |
| ILU(0) factor rel-L2 vs CPU reference | 1.27e-07 | 9.7e-18 | < 1e-3 / 1e-8 |
| IC(0) factor rel-L2 vs CPU reference | 1.64e-08 | 0.0 | < 1e-3 / 1e-8 |
| csrsv2 lower/upper solve rel-L2 | — | ~1e-16 | < 1e-8 |
| Preconditioned-solve worst relative residual | 8.2e-08 | 1.4e-16 | < 1e-3 / 1e-6 |
| Solution rel-L2 vs dense oracle | — | 4.2e-16 | < 1e-5 |
| RHS gradient (adjoint) rel-L2 | 3.9e-08 | 2.9e-16 | < 1e-3 / 1e-5 |

Dense-pattern ILU(0) reconstructs A exactly (no fill dropped); zero/singular-pivot
info matches the CPU reference; 7 refusals pass.

## Feasibility decision (read-only, user-confirmed)

- **rocSPARSE has no general *direct* sparse LU/Cholesky** on any architecture.
  The native options are ILU(0)/IC(0) preconditioners (`csrilu0`/`csric0`,
  general CSR, **wave32-aware** so gfx1013-feasible — unlike the MFMA-blocked
  packed INT4) and banded direct `gtsv`/`gpsv`. The user selected **ILU(0)/IC(0)
  + a bounded preconditioned solve** as the first subset.
- The active-release `libhipsparse` exports `hipsparse[SDCZ]csrilu02/csric02`
  (+`_analysis`/`_bufferSize`/`_numericBoost`), `csrsv2` (+`_analysis`/`_solve`),
  the Info create/destroy and `Xcsrilu02_zeroPivot` (confirmed by `nm -D`).
- The pinned general Torch sparse-solve API rejects ROCm (B02.1 audit), so this
  is a **private-handle ctypes hipSPARSE path** reusing the N2.13 native-
  validation pattern. **No native rebuild, no Torch/library source edit.**
- **Nonsymmetric eig: retain host-LAPACK.** N2.2 production eig is host-LAPACK
  (`hipSOLVER DnXgeev → cgeev/zgeev`); the extensive native Schur prototype
  (N2.3–N2.12) was measured **slower than host LAPACK for small matrices** (N2.6:
  host ~0.2–0.4 ms vs native ~3.75–11.8 ms). The N2 practical review's disposition
  is to retain host-LAPACK and not rework tiny eig. This batch records that
  decision; larger/ill-conditioned native eig remains separate deep work.

## Method

- **Recipe.** Strict no-fill ILU(0) (IKJ) / IC(0) on the input sparsity pattern;
  numeric boost disabled; zero pivot reported, never silently perturbed. An
  independent pure-Python CPU reference (`core.py`) implements the same algorithms
  plus forward/back substitution, a dense Gaussian-elimination oracle and CSR
  transpose — sharing no code with the GPU path.
- **Native path.** ctypes bindings to `libhipsparse` (classic API:
  `hipsparseCreate`/`Destroy`, `CreateMatDescr`, `SetMatType`/`FillMode`/`DiagType`/
  `IndexBase`, `csrilu02`/`csric02`/`csrsv2`). Device memory from torch tensors
  (`data_ptr`). `csrsv2` analysis is **trans-specific** — the adjoint uses
  separate TRANSPOSE-analyzed solvers.
- **Solve.** ILU(0) factors split into unit-lower L and upper U; the preconditioner
  M⁻¹ = U⁻¹L⁻¹ is applied with native `csrsv2`; an outer Richardson iteration
  converges to A⁻¹b. The RHS first gradient (dL/db = A⁻ᵀ dL/dx) uses the adjoint
  preconditioner M⁻ᵀ = L⁻ᵀU⁻ᵀ (TRANSPOSE solves). Non-convergence is reported via
  the residual, never hidden.
- **Guardian.** The qualified PLAN2-01/02 envelope (`guard_nf4.py` reused): SH
  runtime lock, nonce-matched 600 s boot-guard ack, 460 s bounded child,
  `timeout 430` backstop, kernel-log/registry identity (benign governor noise
  filtered, real faults fail), foreign-container refusal, plus a stock-clock gate
  (`pp_dpm_sclk` ceiling ≤2000 MHz).

## Nonsymmetric-eig disposition (recorded, no implementation)

Production nonsymmetric eig **remains host-LAPACK** (N2.2, unchanged). Rationale:
the native small-Schur candidate is slower than host LAPACK at small orders
(N2.6), and the practical review explicitly retains host-LAPACK and deprioritizes
tiny-eig optimization. Native larger/ill-conditioned eig (balancing, two-stage,
convergence-failure contracts) is a separate deep scope, not attempted here. No
eig dispatch/default change.

## The attempt journey (harness/API fixes, not stack defects; all retained)

1. **r1** — staging path bug (`LOCAL_TOOLS` pointed at the wrong package dir).
2. **r2** — classic hipSPARSE naming: handle create/destroy are
   `hipsparseCreate`/`hipsparseDestroy`, not `*Handle`.
3. **r3** — factorization **27/27** (ILU/IC parity, pivot contract, refusals).
4. **r4** — `csrsv2` analysis is trans-specific (solving TRANSPOSE after a
   NON_TRANSPOSE analysis → status 3); fixed with TRANSPOSE-analyzed adjoint
   solvers; also corrected a singular-refusal fixture that was accidentally the
   identity.
5. **r5** — **42/42** (factorization + triangular parity + preconditioned solve +
   RHS gradient + refusals).

No pre-declared bound was relaxed after any run (`tmp/sparse-plan2-03/BOUNDS.md`,
fixed before any board run).

## Boundaries

- Opt-in pure-Python helper (`build_tools/bc250/sparse_plan2_03/`); **no native
  operator, dispatch, default, package or source change**; not installed into any
  runtime image. No general *direct* sparse LU/Cholesky claim (rocSPARSE provides
  ILU/IC preconditioners, not a direct general factorization).
- Bounded owned fixtures (orders ≤256, FP32/FP64, well-conditioned
  diagonally-dominant / SPD). Extreme scaling/conditioning, convergence-failure
  contracts for hard spectra, larger orders, batching, complex dtypes,
  compile/graph integration and performance tuning are separate.
- First derivatives only (RHS gradient via the adjoint); no matrix-value gradient,
  no higher derivatives. The preconditioned solve is iterative (Richardson) with a
  reported residual, not a direct solve.
- Single-board (.104), stock clocks, owned synthetic fixtures only; no user
  application; .105 untouched; host governor/clock/watchdog/kernel not modified.

## Verification

- Workstation: 15 sparse unit tests (CSR helpers, ILU(0)=exact LU on dense
  pattern, IC(0)=exact Cholesky on SPD, pivot detection, dense oracle,
  forward/back substitution, CSR transpose, orchestrator config, guardian reuse);
  full bc250 suite **428 pass**, Black clean.
- On-board: r5 guardian passed (42/42); boot `0d46a6f1…` unchanged; kernel log
  clean (0 benign warnings added); registry unchanged; 0 foreign containers; all
  boot-guards acknowledged; final state 0 containers, r2 kernel intact.

## Reproduction and preservation

Tools: `build_tools/bc250/sparse_plan2_03/` (`core.py`, `ops.py`,
`qualify_sparse.py`, `run_sparse.py`); reuses the qualified `guard_nf4.py` +
PLAN2-01 `bounded_launch.py`/`kernel_noise.py`. From the package directory:

```sh
python3 run_sparse.py --label sparse-r5   # stage + guarded qualification + collect
```

Root steps use `ssh -tt … doas` with the PIN recovered at runtime (never
persisted). The host task directory (`$TASK/sparse-plan2-03`) retains the staged
tools, configs, outputs and per-attempt guardian evidence (r1–r5). Local evidence:
`kernel/artifacts/sparse-plan2-03` (qualification + guardian finals, kernel logs,
preflight, BOUNDS.md, SHA manifest).

## Conclusion

PLAN2-03's sparse capability is **proven**: native general-CSR ILU(0)/IC(0)
factorization with a qualified preconditioned solve and RHS gradient at machine
precision, a preserved refusal/pivot surface, and no native/default/source change
— the scalable native sparse route the dense-LU fallback lacked. Nonsymmetric eig
retains the qualified host-LAPACK default per the recorded evidence. Remaining
PLAN2 items (attention, compile, broader admission, FP16 PEFT) stay optional and
await user direction.
