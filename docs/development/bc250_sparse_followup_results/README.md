# A04/A05 follow-up: sparse backward, timings and lint — 2026-10-16

Installed on .104: Torch `6ed3d12b184a27d21c0fa41d43269ff6bd17f1e0`,
TheRock runtime helpers `4f478cc1d6acd9aa901f011a17d7c79856dccd36`.
[Deployment/source/hash manifest](deployment.json), [38 backward tests](grad-installed.json),
[forward timings](forward-timings.json), [training timings](training-timings.json),
[original failures](baseline.json), [lint output](lint-final.log).

This closes the requested first-order backward, bounded performance-measurement
and lint-initialization gaps before returning to A06. It does not turn sparse
execution into an automatic training optimization.

## What was fixed

- Raw `_cslt_compress` and `_cslt_sparse_mm` now have native gfx1013 first-order
  derivatives for FP32/FP16/BF16. Compression scatters gradients only into the
  retained slots recorded by that forward pack, including retained zero values.
  Sparse matmul differentiates retained values, dense input, bias and scalar or
  per-row alpha; ordinary/transposed results are handled.
- Public `to_sparse_semi_structured` now preserves the gradient edge to a
  trainable dense source weight on gfx1013. Its gradient is projected onto the
  current retained mask. Previously conversion created a separate leaf: the
  baseline collector's public-weight row returned without an exception but
  explicitly recorded `w_grad=false`; it was not a correctness pass.
- A row-sparse weight need not remain row-sparse after transposition. When a
  public-wrapper gradient needs that missing transpose pack, reconstruct the
  weight on the GPU and use dense GPU matmul. No pruning of the transpose or CPU
  matrix-arithmetic fallback is used. Raw backward also uses dense GPU buffers.
- Saved absent optional tensors may arrive as undefined tensors inside an
  engaged optional. Both `has_value()` and `defined()` are now checked.
- Lint tools initialize successfully in the local source workspace. The offline
  build image had neither initialized tools nor a Git source checkout; that was
  a tooling setup failure, not a numerical failure. Configured binaries were
  SHA256-verified, dependencies installed in isolated uv caches, and source
  formatting/newline issues fixed. No board runtime packages were changed for lint.

## Tested scope

Candidate and installed38 device-generic TestCase tests pass CPU references:
18 raw cases (three float types, ordinary/transposed output, no/scalar/vector
alpha),16 public cases (FP16/BF16,32x32 and64x32 weights, frozen/trainable source,
matmul/linear including bias), three independent-gradient groups with five
subsets each, and one retained-zero group. Custom streams and a column-contiguous
dense operand are covered. No tolerance was relaxed after a numerical failure.

Installed regressions pass20 A04 forward tests/120 small matmuls,69 A05 tests,
seven core reference groups and30 convolution output/gradient configurations.
Additional timings CPU-check12 forward and12 forward/backward fixtures before
measurement.15 total GPU/diagnostic guards, including retained failures, ACK
within600s; every before/after dmesg pair is identical, SHA
53f83f7bbee6c060e3153e7ae95a4ff069bf9627113591c39cdddb9ab5f4b973.

The final native build is retained container `bc250-sparse-followup-build-r4`,
image28ad36e6, log `WS/logs/sparse-followup-build-r4.log`. Only editable pip
build/codegen was used, with ccache20GiB,12 jobs, one link job, native gfx1013 and
no GPU devices. Adding native schemas caused a broad generated-header rebuild.
Initial concrete-size codegen, C++ operator/ADL compile, early-import and absent-
optional failures were retained and corrected. Final candidate/installed results
use the repaired source. No running agent build/test/container/guard remains.

## Measured performance and resulting policy

The table reports median eager GPU-event microseconds per forward call, using
six alternating sparse/dense rounds of ten calls after warmup. Wall times and
all samples are retained in JSON. Sparse multiplication reuses an existing pack;
compression is measured separately. These are operator envelopes including host
submission gaps, not captured kernel-only throughput.

| dtype | M,N,K | prepacked sparse us | dense us | compression us | dense/sparse |
| --- | --- | ---: | ---: | ---: | ---: |
| float32 | 32,16,32 | 14.66 | 17.46 | 8.17 | 1.190 |
| float32 | 128,64,128 | 40.11 | 39.64 | 8.93 | 0.988 |
| float32 | 256,128,256 | 123.67 | 73.62 | 11.20 | 0.595 |
| float32 | 512,64,512 | 238.99 | 131.84 | 22.89 | 0.552 |
| float16 | 32,16,32 | 12.61 | 19.84 | 9.81 | 1.573 |
| float16 | 128,64,128 | 35.51 | 41.19 | 10.98 | 1.160 |
| float16 | 256,128,256 | 108.29 | 30.21 | 13.30 | 0.279 |
| float16 | 512,64,512 | 208.99 | 137.13 | 25.55 | 0.656 |
| bfloat16 | 32,16,32 | 12.62 | 20.00 | 10.10 | 1.585 |
| bfloat16 | 128,64,128 | 35.95 | 35.82 | 11.04 | 0.996 |
| bfloat16 | 256,128,256 | 111.84 | 35.50 | 13.32 | 0.317 |
| bfloat16 | 512,64,512 | 216.35 | 115.55 | 25.72 | 0.534 |

Prepacked forward wins in some small fixtures (up to1.585x here), but adding
compression erases several wins and larger fixtures lose (dense/sparse down to
0.279). The12 training fixtures use M,N,K32,16,32 or128,64,128 across the same
three types, with either frozen weights/reused pack or trainable weights/
recompression. Dense training applies the same retained mask when weights train.
Six alternating rounds of five forward+autograd.grad calls measure the full
operator envelope, including metadata checks/host synchronization and dense GPU
backward buffers. Sparse training is1.92–3.36x slower in these fixtures
(dense/sparse0.298–0.522). Both paths pass independent CPU gradient references.

**Policy:** keep default dense rocBLAS and normal training unchanged. Structured
sparsity remains explicit; there is no hardware sparse matrix accelerator on this
board. These small fixtures are neither universal timing bounds nor an application
speedup/convergence claim. They provide a correctness-first training baseline for
future optimization, not a reason to switch existing LoRA scripts automatically.

## Lint workflow

The following source-only workflow now succeeds locally:

```bash
PYTORCH_LINT_SKIP_CODEGEN=1 uv run --no-project --python 3.12 --with spin==0.18 \
  spin lint -- --take=PYFMT,RUFF,FLAKE8,CLANGFORMAT,NEWLINE,INCLUDE,NATIVEFUNCTIONS,TESTOWNERS,TEST_HAS_MAIN,IMPORT_LINTER,SET_LINTER,DOCSTRING_LINTER <affected-files>
```

The new opt-in environment flag prevents spin's normal version/type-stub
regeneration, keeping all codegen inside the permitted pip build. Default spin
behavior is unchanged. Exact file lists are in lint-final.log. The required
`lintrunner -a` was also run with this same target set and passed. This is targeted
configured static/format testing, not a claim of full-project type analysis or a
complete CI run. Optional absent .lintrunner.private.toml emits its normal notice.
[Verified tool provenance](lint-tools.json): spin0.18, lintrunner0.12.7, configured
clang-format/clang-tidy19.1.4 and actionlint1.7.7. Cached tools/artifacts were
preserved; the initial slow partial download is retained in ignored agent_space.
[Formatting verification](format-verification.json) confirms the A05 function/
class ASTs are unchanged; the formatter canonicalized import order. A04 test
formatting/owner metadata and a version-binding line wrap are also included.

## Deployment and limits

Under the existing exclusive runtime lock, current/torch atomically switches from
WS/pytorch-a04 to the independent WS/pytorch-sparse-followup tree, still mounted
at /workspace/pytorch-work. `bc250-torch`/`bc250-train` therefore use the repaired
sparse paths when explicitly requested. Original A05 and older Torch trees remain
intact; recovery link/runtime metadata are in WS/recovery/sparse-followup-before.
Failed installed gates would restore that link/metadata.31 protected hashes are
unchanged, including original Torch, ROCm DSOs/DB/kpack, frozen fallback, original
LoRA trainer/helper and consumer bundle. New libtorch_hip SHA
8ffd32eeaa6548ef896a5f56773255a26a985398a4a7bf811463bcb1dc043b90.

WS=/home/nik/bc250work/rocm-10.2-gfx1013; raw WS/logs/sparse-followup-* and local
ignored kernel/artifacts/sparse-followup/. SOURCE_SPARSE_FOLLOWUP.json is in
WS/current. Full local Torch Git and incremental patch0005/bundle on6181d00 are
retained; reverse-apply and bundle verification pass. The older three-file SDPA
bundle reconstruction gap remains separate. Disk8.7GiB free; preserve existing
sources/builds/caches and avoid unnecessary full-tree copies.

First-order floating operations on valid native packs are the tested scope.
Public wrappers qualify FP16/BF16; raw operations also qualify FP32. Integer
backward, higher-order/forward-mode AD, compile/export/graphs and broad shapes,
numerics, concurrency or sustained stability remain outside this qualification.
Sparse arithmetic is on the GPU; metadata validation can synchronize with the
host. Pack arithmetic/forged metadata and arbitrary dense inputs without valid
1:2/2:4 sparsity are not supported. Kernel/boot/BIOS/.105/user data/therock/src/,
SDMA0/heap-off/math SDPA and r2 fallback are unchanged. Historical Python CPU
segfault remains unresolved and distinct from fixed VM-heap/CSR corruption.

Exact next **A06** remains the recorded gated-delta/FLA reference profile and one
native scalar/wave32 substage with independent output/Q/K/V/gate gradients.
A06–A09 precede N2.7; K6 stays completed.
