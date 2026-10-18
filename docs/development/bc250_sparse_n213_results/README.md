# N2.13 sparse triangular solve coverage and exact pivots — 2026-10-18

Completed and installed on .104. Generic CSR vector/multi-RHS and BSR block
triangular solves pass the bounded FP32/FP64/complex64/complex128 CPU-reference
matrix below. A demonstrated Torch singular-result defect is corrected: exact
zero or missing diagonal pivots now produce NaN in the corresponding output rows.
The existing native solver path remains. This is not general sparse factorization
or torch.sparse.spsolve support, and it is not a performance improvement claim.

Native Torchdc8fea0 (implementationa434d03 plus2D guard); final test qualification
2b449c4. TheRock buildbbf556448, controller/sealing39ea58574. Active
WS/pytorch-sparse-n213; old WS/pytorch-profiler-order retained independently.
[Final identities](final.json), [installed cohort](installed.json),
[summary](summary.json). Patch0010 is a three-commit mbox on qualified Torchda45369.
Full local Git remains authoritative; historical SDPA reconstruction gaps remain.

## Audit and reproduced defect

Earlier P2 evidence covers25 Torch sparse cases plus16 direct legacy real csrsv2
cases. It does not establish current generic SpSV/SpSM complex/index/lifecycle
coverage. The pinned Torch CSR path calls generic hipSPARSE SpSV for one RHS,
SpSM for multiple RHS; BSR uses legacy block vector/matrix APIs. All route to
existing native rocSPARSE, with native gfx1013 and safe runtime defaults.

The initial256 nonsingular cases pass before modification. Singular/empty probes
then fail16/32; the expanded four-dtype probe fails39/80. rocSPARSE avoids zero
pivot division and can leave finite placeholder values. Torch's BSR pre-solve
pivot query covers structural block failures but does not catch every numeric
scalar pivot. Generic CSR exposes no equivalent public hipSPARSE pivot query in
this consumer. A finite-looking result can therefore escape the Torch wrapper.
The numerical payload of these invalid singular results is unspecified; do not
rely on their exact old values or a fixed count across arbitrary reruns.

## Correction and numerical boundary

After the native solve, a ROCm-only2D wrapper launches one GPU diagnostic for
non-unit, nonempty CSR/BSR solves. Each thread binary-searches the sorted compressed
indices for its scalar row's diagonal block and checks that diagonal element.
Missing or exact signed-zero pivots mark that row across all RHS columns as NaN.
Bit comparisons on both complex components avoid treating tiny nonzero bit
patterns as exact zero. There is no host scalar read, dense conversion, extra
workspace allocation, CPU fallback, pivot perturbation or new epsilon policy.

The kernel follows the actual current stream, 32/64-bit compressed metadata and
output/value strides, including column-major blocks and padded outputs. Existing
BSR backend indices are internally downcast to32-bit by Torch; accepting public
int64 metadata here does not qualify giant native64-bit BSR indexing. Other rows
of a singular solution remain unspecified; no rank/conditioning estimate or
near-zero detection is added. Existing empty-A NaN behavior is retained even
with unitriangular=True. Unit-diagonal solves bypass the new check. Batched/other
dimensional paths are unchanged/unqualified. Direct hipSPARSE/rocSPARSE APIs are
unchanged and do not inherit this Torch result-marking policy.

Nonsingular work now includes one additional diagonal-search GPU launch; it does
not write healthy output. Throughput for large systems remains unqualified. This
scope does not establish sparse triangular autograd, compile/graphs, arbitrary
concurrency, all formats or every size/value/conditioning contract.

## Verification

Candidate and installed runtime each pass:

| Group | Count | Evidence boundary |
| --- | ---: | --- |
| Public matrix |256 |four dtypes, CSR/BSR, public32/64 indices,1/3 RHS, upper/lower, transpose, unit/nonunit; B layouts, strided guarded output, column-major blocks, private stream; n17/18 |
| Pivot/empty |80 |missing scalar diagonal, stored zero, missing block, empty and legacy empty-unit; final test requires NaN at pivot, strided guards, unchanged input and private stream |
| Larger references |32 |CSR65/257, BSR66/258 with block3, both RHS counts/all four dtypes; bounded well-conditioned CPU references |
| Implicit unit |64 |omitted diagonal with nonempty strict triangle; upper/lower/transpose/vector/multi-RHS; no false NaN |
| Actual kernel trace |1 |one native sparse_triangular_pivot_mask_kernel in a public complex CSR solve |
| Direct generic |96 fixtures/192 solves |four dtypes, native CSR32/64, N/T/adjoint, SpSV/SpSM, host/device alpha, padded output, private stream; two submissions reuse one analysis per fixture |
| Retained P2 sparse |41 |25 ordinary COO/CSR/gradient cases and16 legacy real triangular cases |
| Core/profiler |7 core +one ordering process |output/gradient GEMM/attention/norm/FFT, exact RNG/consumers; CPU-first profiler still records correlated kernels/copies with exact gradients |

The433 new public groups use CPU double/complex128 solves; value/residual bounds
3e-5 for float32/complex64 and2e-12 for float64/complex128. Inputs and output
sentinels remain unchanged where promised. Native direct tests independently
compute CPU residuals with the same precision bounds and hold handles/descriptors/
workspace/scalars until the pinned stream completes, then destroy in reverse
order. They test valid inputs, not a new direct-native singular policy.

The final pivot test was strengthened after the native build to exercise actual
writes into padded outputs on a private stream. That test-only revision is
recorded separately from native GIT_REVISIONdc8fea0; native binaries and numerical
bounds were unchanged. Only that group was rerun; other unchanged groups and
verified binaries retained their completed evidence. Earlier candidate manifests
are preserved. Targeted spin and lintrunner -a checks pass, not full CI.

All22 guards acknowledge600s:20 successful process gates, two honestly retained
pre-fix numerical failures. Kernel logs are byte-identical over the entire batch;
no GPU fault/reboot.121 prior/source/runtime identities are preserved, with the
intentionally changed active runtime metadata/GPU-library aliases checked through
recovery paths. Only libtorch_hip.so changes among native DSOs; the CPU profiler
fix remains byte-identical. No broad model/convergence/performance claim.

## Reproduction, storage and recovery

WS=/home/nik/bc250work/rocm-10.2-gfx1013. Work locally and transfer staged source;
helpers/build_sparse_n213_on_host.py makes independent source/object/DSO copies,
with the qualified sparse-followup/third_party mounted read-only. Required
out-of-place HIP generation precedes pip editable/no-build-isolation build;
ccache20GiB, jobs12/link1, native gfx1013 and existing safe flags remain.
[Build command](build-command.json), compressed full build.log.gz and generated
HIP hashes in final.json retain exact inputs/output provenance. No cache reset
or original source/build removal; no metadata-only Torch rebuild.

Run under the existing exclusive update_dev_prefix.py lock:
qualify_sparse_n213_on_host.py --phase candidate, then --phase install;
finalize_sparse_n213_on_host.py seals passed gates and identities. Never nest an
ordinary launcher's shared lock. Exact guarded runtime commands are retained.

Recovery WS/recovery/sparse-n213-before holds exact old runtime.json, original
Torch link text and a link to the complete old runtime. Automatic failure rollback
restores both the link and metadata; manual rollback must do the same atomically
under the exclusive lock. SOURCE_SPARSE_N213.json in WS/current records the active
cohort. The old ROCm DSOs/packs, current/previous/rocm links, frozen release,
runtime image, .105, BIOS/kernel/boot and r2 fallback remain. Only the development
Torch link and corresponding runtime metadata change.

Raw WS/logs/sparse-n213-*; ignored local kernel/artifacts/sparse-n213. No running
agent job/container/guard; free~7.23GiB. Next preservation baseline is
WS/logs/sparse-n213-qualification/final.json:121 preserved identities plus current
candidate_sha256/runtime_sha256. Earlier active GPU alias hashes are historical.
Historical Python CPU segfault remains unresolved and separate from fixed
VM-heap/CSR corruption; these pivot failures are numerical API behavior, not that
memory defect. No general sparse solve B02, sparse solve autograd or universal
performance/correctness claim.

Next N2.14: audit existing P2 FFT30, then qualify bounded batched/strided complex
plans and odd-length real transforms with CPU outputs/gradients. Preserve the
existing rocFFT AOT cache; BF16 promotion B01 remains a separate queue item.
