# A04: native Torch hipSPARSELt integration — 2026-10-16

**Later follow-up:** native first-order sparse backward and targeted lint
initialization/checks are now qualified; bounded forward/training timings are
recorded in the [current follow-up report](../bc250_sparse_followup_results/README.md).
The original milestone evidence/limitations below remain historical.

Installed on .104: Torch `8cdfcc8e0feb3689a1ec1d292539af083b13ea94`,
TheRock helpers `b5879aefa7990928270645676e66cc40d972de6b`.
ROCm libraries remain `fcf7a7142af8c258164e106e66bf893c7f563d15`;
ROCr329eaf7, Triton ee65ff3b, image5df0a58 and running r3 kernel remain unchanged.
[Deployment/source/hash manifest](deployment.json), [baseline](baseline.json),
[installed sparse results](sparse-installed.json), [core](core-installed.json),
[attention smoke](attention-installed.json).

The previous Torch cache recorded `hipsparselt_DIR-NOTFOUND` despite the package
now being installed. Existing CMake discovery/version gates already work after a
fresh candidate configure; no package-discovery source workaround was necessary.
The Torch architecture list excluded gfx1013, FP32 was disabled on HIP, and the
Python version/search bindings were CUDA-only. `_has_cusparselt` was true on every
ROCm build even when the operation was compiled out. Baseline compression failed
while `is_available()` incorrectly returned true.

The patch enables the native scalar backend and modern Python bindings, reports
availability from compiled support, initializes returned search fields, and checks
native pack capacity/layout/device/dtype/bias/alpha contracts before execution.
FP32 bias type is explicitly described to hipSPARSELt. Version209 follows the
installed 0.2.9 header. The native library itself was already qualified by N1.4–7.

## Bounded GPU evidence

Both separate candidate and installed runtime pass20 Torch TestCase tests:

- 16 pack/matmul combinations: FP32/FP16/BF16/INT8, geometries7x9x12 and32x16x32,
  each row/column-contiguous dense input. Independent values/index-byte decoding
  reconstructs the original valid sparse matrix exactly.
- 120 small matmul executions across those combinations: plain, GPU scalar
  alpha+FP32 bias, GPU per-row FP32 alpha+bias, typed floating bias, ordinary and
  transposed results, custom streams. CPU double references; INT8 round-even and
  saturation reference. Integer/dyadic fixtures intentionally avoid rounding noise.
- FP16/BF16 `to_sparse_semi_structured` plus public `torch.mm` references.
- Modern search returns `(algorithm=0, split_k=1, split_mode=0, algorithm_count=1)`;
  deprecated search returns0 and executes correctly.14 unsupported-call refusals.

Installed regressions:30 convolution output/gradient configurations, seven core
reference groups (dense GEMM, math SDPA, normalization, FFT, RNG), and one tiled
SDPA output/QKV-gradient smoke configuration. Eight GPU/diagnostic guards total,
including retained harness/test failures, all ACKed within600s; dmesg identical
SHA53f83f7bbee6c060e3153e7ae95a4ff069bf9627113591c39cdddb9ab5f4b973.
18 protected runtime/database/original-Torch/fallback hashes remain unchanged.

Raw logs: `.104 WS/logs/torch-a04-*`; local ignored
`kernel/artifacts/torch-a04/`. WS is `/home/nik/bc250work/rocm-10.2-gfx1013`.
Builds used the existing ccache20GiB/jobs12/link1/native-gfx1013 configuration and
only `python -m pip install -e . -v --no-build-isolation`, without GPU access.
Final build: retained container `bc250-a04-torch-build-r4`, log
`WS/logs/torch-a04-build-r4.log`; no active agent jobs remain.

Retained failures: initial test launcher used an unmounted helper path; next
matrix incorrectly expected CPU scalar alpha acceptance and version207. Torch's
generated wrapper requires alpha on the same GPU; installed headers specify209.
Tests now assert CPU scalar rejection. Earlier build r3 used raw CUDA source with
an unchanged CMake cache, so HIPify was skipped; refreshing candidate configure
inside pip fixed it. Build r2 also hit a changed-live-script tail after pip success;
all logs/containers retained. No GPU fault or numerical tolerance relaxation.
`spin lint` could not initialize its offline environment; the required
`lintrunner -a` attempt also failed on missing initialization. No lint pass claimed.
Python compilation, shell syntax and Git whitespace checks pass.

## Deployment and limits

Under the exclusive runtime lock, `current/torch` atomically switches from
`WS/pytorch-work` to the independently copied/built `WS/pytorch-a04`. Container
mount destination stays `/workspace/pytorch-work`. Daily `bc250-torch` and
`bc250-train` use it automatically. The complete original tree is unchanged;
recovery link/runtime metadata are in `WS/recovery/torch-a04-before`. Failed
installed gates would restore that link and metadata. Frozen stack and r2 boot
fallback, BIOS, .105, user data and `therock/src/` remain untouched.

Use valid prestructured sparsity: FP32 is1:2, FP16/BF16/INT8 are2:4. Compression
is not magnitude pruning. Public semi-structured wrappers were checked only for
FP16/BF16. The direct native path produces same-dtype outputs; widened INT8/FP8,
split-K and multiple streams per operation are unsupported. Scalar alpha must be
a GPU tensor; row-vector alpha is contiguous GPU FP32. Bias is FP32 or matching
floating input dtype, one value per sparse row. Packs contain no provenance;
shape checks cannot validate arbitrary forged metadata. Sparse autograd, broad
shapes/numerics, compiler/graph capture, sustained stability and speedups remain
unqualified. Scalar GPU execution provides no hardware sparse-matrix acceleration.
Default dense BLAS remains rocBLAS. Other prefixes/independent venvs are unchanged.

Full patched Torch Git remains local. Incremental A04 patch and Git bundle are
retained locally and at `WS/helpers/torch-a04.{patch,bundle}`. The old P2 patch
bundle still lacks three later SDPA files; adding A04 does not close that separate
reconstruction gap. Historical Python CPU segfault remains unresolved and distinct
from the fixed VM-heap/CSR corruption.

Exact next **A05**: inspect the existing Qwen3.5 causal-conv/loss helpers and
recorded51 operator/model/resume gates; expose a reusable opt-in API and add
trainable convolution gradients with CPU/checkpoint/AMP references. A05–A09 remain
before N2.7; K6 stays completed.
