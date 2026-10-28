# Using bc250-torch for training and persistent files

> **Historical snapshot notice:** this file preserves a previous milestone.
> Read [flow.md](../../flow.md) and the
> [current memory inventory](bc250_memory_inventory.md) before acting on its
> active-release claims or unchecked tasks. Newer qualified scopes supersede
> older unsupported/pending rows; retain the original evidence.

The reviewed daily-use release now supplies `bc250-train train.py`, which selects
the independently frozen Transformers/PEFT/Accelerate packages. The original
developer venv remains available. See [bc250_release_20261013.md](bc250_release_20261013.md).

`bc250-torch` runs the validated Torch/ROCm image through rootful Podman on
`.104`. Python itself runs as `nik` (UID/GID 1000). The ROCm and Torch runtime
files are read-only; your project and data files use host bind mounts.

## Local CPU distributed execution (B03.1)

The current development Torch build includes native Gloo process groups and
libuv TCPStore. Two local CPU ranks pass collective/first-gradient and DDP
FP32/FP64 SGD-momentum/AdamW checks against independent references. This is
explicit functionality; existing scripts are not distributed automatically.
For an owned CPU DDP script that initializes `backend="gloo"`:

```bash
bc250-torch --env GLOO_SOCKET_IFNAME=lo -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 /path/to/owned_cpu_ddp.py
```

Keep this qualified model/tensor path on CPU and use bounded thread counts.
GPU Gloo qualification is described below; RCCL, RPC, non-loopback networks,
multi-board scaling and distributed checkpoint recovery remain separate. Frozen old prefixes retain
their previous build. [Evidence, boundaries and rollback](bc250_distributed_b031_results/README.md).

## Gloo GPU tensors and DDP on one board (B03.2)

The development Torch's Gloo backend also passes bounded GPU tensor collectives,
first gradients and FP32/FP64 GPU DDP reference tests. Two processes share the
single BC-250 GPU; communication/reduction uses pinned CPU staging while the
model runs on HIP. Explicitly select `backend="gloo"`, `cuda:0` in both ranks,
and `DistributedDataParallel(model, device_ids=[0])` in an owned script:

```bash
bc250-torch --env GLOO_SOCKET_IFNAME=lo --env GLOO_DEVICE_TRANSPORT=TCP \
  -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  /path/to/owned_gpu_ddp.py
```

Use bounded thread counts and allow memory for both processes. No throughput or
multi-GPU scaling improvement is established. GPU send/recv, RPC, BF16 DDP,
distributed checkpoint recovery and other networks/devices remain unqualified;
RCCL remains off in the daily profile; the explicit B03.8 profile below is separate. [Exact tested APIs and limits](bc250_distributed_b032_results/README.md).

## Opt-in RCCL on .104 (B03.8)

The daily runtime remains NCCL-off/Gloo-on. The separate native gfx1013 profile
is explicitly selected per invocation:

```bash
bc250-torch --runtime-profile rccl-single /path/to/owned_script.py
bc250-train --runtime-profile rccl-single /path/to/owned_script.py
```

The script must explicitly initialize `backend="nccl"` with rank0/world_size1
and `cuda:0`. The frozen training consumers remain available through
`bc250-train`; no application is automatically converted to distributed training.
Omit the option, or use `--runtime-profile daily`, to retain daily behavior.

The launcher verifies the packaged RCCL and native Torch identities, mounts
RCCL read-only at `/workspace/rccl`, and holds the daily runtime lock. It fixes
`NCCL_WIN_ENABLE=0`, both CUMEM settings0, P2P/IB disabled, socket interface`lo`,
and asynchronous error handling1. Conflicting explicit overrides are refused.
Hash verification adds startup I/O; this is a functionality profile, with no
speedup claim. Update or unregister this dependent profile only under the
**daily** prefix's exclusive update lock, not an independent profile lock.

Only one GPU/rank is qualified. The375-object filtered library is not a full
RCCL backend; numeric FP8, symmetric device execution, actual offload-payload
relocation, multi-device transport and scaling remain unqualified/unsupported
as recorded. B05.3 updates the opt-in Torch cohort with the native INT8 layout fix; the same
explicit selection/settings apply. [Matched update/rollback](bc250_quant_b053_results/README.md).
`.105` does not receive this profile.
[Acceptance, identities and rollback](bc250_rccl_b038_results/README.md).

## Experimental weight-only INT8 recipe (B05.1, .104)

The .104 runtime now installs the optional `bc250_quant 0.1.0` package (B05.6).
`from bc250_quant import pack, linear, FrozenW8Linear` works in scripts run through
`bc250-torch` or `bc250-train`, including the explicit `rccl-single` profile.
It provides bounded GPU forward/input/bias first gradients. Import/conversion is
explicit; ordinary Torch models are not automatically converted. The earlier
`WS/helpers/bc250_quant_b051.py` interface remains available unchanged. It is not packed INT4,
NF4, a TorchAO/bitsandbytes backend or a complete QLoRA recipe.

B05.2 fixes native strided-activation/FP32-scale wrong results in the daily
Torch build automatically. B05.3 also fixes the `rccl-single` native cohort.
[Matched profile qualification](bc250_quant_b053_results/README.md). The prototype normalizes layouts in both profiles and supplies its
own input derivative; native backward remains unsupported. No quantization/model
routing is enabled automatically.
[Native fix, qualification and rollback](bc250_quant_b052_results/README.md). Two small forward fixtures were about3.2× slower than resident
dense FP32, while fresh packed weight payload was about1.97× smaller than FP16.
No model speed/quality or total-memory improvement is established.
[Exact API, bounds, CPU references, timings and limitations](bc250_quant_b051_results/README.md).

B05.4 updates this explicit helper with one normal-path validation scalar read
instead of two, preserving finite checks/error priority and first gradients.
Three FP32 paired fixtures show1.075–1.145× forward and1.029–1.080× forward+
backward old/new latency ratios across daily/RCCL cohorts. These are helper-only
measurements, not training/model gains or proof of beating dense GEMM.
[Measurements, error contracts and helper rollback](bc250_quant_b054_results/README.md).

### Explicit frozen-W8 module and checkpoints (B05.5)

`FrozenW8Linear` is installed in the public optional package on .104 (B05.6).
Run an owned script with `bc250-torch script.py` or `bc250-train script.py`;
no helper-directory `PYTHONPATH` is needed:

```python
import torch
from bc250_quant import FrozenW8Linear

source = torch.nn.Linear(128, 64, device="cuda", dtype=torch.bfloat16)
source.weight.requires_grad_(False)  # Explicitly freeze before conversion.
layer = FrozenW8Linear.from_float(source)
del source  # Conversion does not remove the caller's floating source.
x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
layer(x).float().square().mean().backward()
torch.save(layer.cpu().state_dict(), "frozen-w8.pt")
restored = FrozenW8Linear.from_state_dict(
    torch.load("frozen-w8.pt", map_location="cpu", weights_only=True), device="cuda"
)
```

Convert and choose device/dtype **before constructing an optimizer**. Bias values
and trainability are copied; the floating base is not retained by the module.
Only bias is a possible module parameter; INT8 weights and FP32 scales are
persistent buffers. FP16/BF16/FP32 moves keep scales FP32 without rounding them
through the compute dtype. Forward requires matching input dtype even under AMP.
CPU storage/load is allowed; conversion and forward require native gfx1013 GPU.

State v1 records shape, compute dtype and bias policy. Loading requires complete
matching metadata/payload even with `strict=False`; `assign=True` is refused.
Invalid own-module payload is validated before copying its tensors. Loading a
larger parent model is not globally transactional. As with ordinary modules,
train/eval mode and the optimizer are managed separately by the caller.
[Contracts, tested bounds, checkpoint resume and limitations](bc250_quant_b055_results/README.md).

B05.6 preserves B05.5 state v1 in both directions: existing helper checkpoints
load through the public API, and public state loads through the retained helper.
The distribution copies the qualified core exactly and changes only the module's
internal import to a relative package import. Native Torch backward/INT4 limits
and helper bounds remain. This package does not install TorchAO/bitsandbytes or
provide full QLoRA/model-quality/speed guarantees.
[Package identities, acceptance and rollback](bc250_quant_b056_results/README.md).

## ONNX export and explicit CPU inference (B06.1, .104)

The runtime image now provides pinned ONNX1.23.2, ONNX Script0.7.2,
ONNX IR1.1.0 and ONNX Runtime1.31.0. They are available through `bc250-torch`
and `bc250-train`, including the explicit `rccl-single` profile.
The modern `dynamo=True` exporter is qualified for owned FP32 dense, CNN and
gated/normalization fixtures, exporting from both CPU and native gfx1013 tensors.

```python
import numpy as np
import onnxruntime as ort
import torch

model = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.ReLU(),
                            torch.nn.Linear(12, 4)).eval().cuda()
x = torch.randn(3, 8, device="cuda")
program = torch.onnx.export(model, (x,), dynamo=True, opset_version=18,
                            input_names=["x"], output_names=["y"])
program.save("owned-example.onnx", external_data=False)
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession("owned-example.onnx", sess_options=options,
                               providers=["CPUExecutionProvider"])
session.disable_fallback()
y_cpu = session.run(["y"], {"x": x.cpu().numpy()})[0]
y_gpu = torch.from_numpy(np.array(y_cpu, copy=True)).cuda()
```

**ONNX Runtime inference here runs on CPU.** The example explicitly transfers
results back to Torch GPU. It does not connect autograd through ONNX Runtime;
subsequent GPU operations can compute gradients with respect to a new Torch leaf.
There is no installed/qualified ROCm or MIGraphX execution provider. Always check
`ort.get_available_providers()` before requiring another provider; naming a missing
provider alone can allow ONNX Runtime to fall back to CPU. The retired ROCm EP
is not a flag that can enable gfx1013 in this runtime.

Tested dynamic batches1/3/4 use Torch export bounds1–4. ONNX symbolic dimensions
do not enforce those Torch bounds at inference. Other dtypes/models/operators,
ONNX training/autograd, GPU execution providers and quantized W8 export remain
unqualified. Existing Torch GPU math/backward and optional W8/FFT/sparse helpers
are preserved. [Pinned packages, reference tests, applicability and recovery](bc250_onnx_b061_results/README.md).

## Explicit MIGraphX GPU inference (B06.8, .104)

A versioned FP32 inference SDK is installed separately from the daily runtime:

```bash
bc250-migraphx --sdk-info
bc250-migraphx /path/to/owned-inference.py
bc250-migraphx --runtime-profile rccl-single /path/to/owned-inference.py
```

Use `import migraphx` inside that script and explicitly compile supported ONNX
programs for `migraphx.get_target("gpu")`. The launcher verifies the registered
SDK and selected runtime identities, mounts SDK read-only and uses the existing
CPython 3.12 image. It accepts ordinary launcher mounts/environment/Python options,
including external script paths containing spaces. Plain Torch launchers keep
their defaults; ONNX Runtime remains CPU-only. Static FP32 IR10/opset18 fixture
coverage does not establish every model/operator/dtype, graph autograd or speed.
[Usage, exact scope, identities and unregister/re-register recovery](bc250_migraphx_b068_results/README.md).

## Exact-layout BF16 GEMM tuning (N3.2, 2026-10-19)

The current mutable development prefix automatically supplies one qualified
rocBLAS solution override through `bc250-torch` and `bc250-train`, including
scripts that invoke those launchers. No trainer edit is needed. The preset
matches BF16 input/output, FP32 compute, beta0, batch1, rocBLAS N,T
M2048/N1024/K6144, lda2048/ldb1024/ldc2048. This is the backward projection
layout found in the synthetic 1024-token Qwen3.5 LoRA audit. Other shapes keep
their existing selection; math SDPA and the default rocBLAS backend remain.
Direct Python environments outside these launchers do not automatically get
this prefix preset. Frozen previous releases have no preset metadata.

The launcher verifies the CSV, rocBLAS DSO and gfx1013 library assets before
using it. Missing/changed identities disable automatic tuning with a message.
An explicit `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` takes precedence. To keep the
original selection, set this on the **outer** command:

```bash
BC250_GEMM_TUNING=0 bc250-train train.py
BC250_GEMM_TUNING=0 ~/finetune-n7/dlvenv/bin/python ~/finetune-n7/scripts/train_bc250.py --resume /path/to/checkpoint
```

The selected kernel changes floating-point reduction order. In the bounded
1024-token synthetic model test, loss and RNG matched exactly; worst adapter
gradient relative L2 difference was 2.03%, below the existing 4% contract.
Each worker's AdamW update/state also passed an independent CPU replay.
Existing checkpoints can still be resumed, but changing this setting does
not promise bitwise continuation. Warm synthetic whole-step timing improved
1.52–1.56x; sustained dataset throughput and convergence remain unqualified.
See the [N3.2 report](bc250_gemm_n32_results/README.md) for CPU GEMM references,
cold-loader repair, installed checks and reversible recovery.

## Current-preset checkpoint qualification (N3.5)

Owned synthetic Qwen3.5-2B tests now cover1024 tokens/rank16/BF16/math SDPA,
accumulation16 and an optimizer-boundary checkpoint. Fresh-process continuation
matches all372 adapter parameters/gradients, AdamW/scheduler state, losses and
CPU/GPU/Python RNG exactly. Independent accumulated-gradient/CPU AdamW checks
also pass. This qualifies the tested helper/runtime case; actual deployed CLI
coverage at that geometry is next. It does not promise bitwise continuation
across changed presets/backends or resume halfway through accumulation.
[Evidence and limits](bc250_training_n35_results/README.md).

## Dense INT8 matmul (A08, 2026-10-18)

The selected .104 ROCm prefix now supports `torch._int_mm` through native
hipBLASLt, producing INT32 output from signed INT8 matrices. No launcher flag or
Torch rebuild is required. The pinned Torch API requires M>16, positive K/N
multiples of8, and contiguous INT32 output. Row/column/strided inputs and `out=`
passed bounded exact-reference tests. Arithmetic wraps modulo2^32 on overflow.

```python
import torch

a = torch.randint(-128, 128, (32, 64), device="cuda", dtype=torch.int8)
b = torch.randint(-128, 128, (64, 32), device="cuda", dtype=torch.int8)
result = torch._int_mm(a, b)
```

This supplies an integer primitive. It does not quantize models, install a
QLoRA/low-bit package, or automatically speed up BF16 LoRA. Mixed bounded timings,
raw Lt scalar/epilogue contracts and recovery are in the
[A08 report](bc250_dense_int8_a08_results/README.md). Packed formats, quantization
recipes, integer autograd and compile/graph integration remain unqualified.

## GPU profiler (A09, 2026-10-18)

The selected .104 Torch build now supports real GPU tracing through Kineto and
rocprofiler-sdk. Use `torch.profiler.profile` with CPU and CUDA activities (ROCm
uses the CUDA API name), then export a Chrome trace to your persistent project
folder. GPU kernels, copies and CPU correlations are recorded; `profile_memory`
adds allocation/free events. This applies to callers of this Torch runtime,
including bc250-torch/bc250-train, rather than a single LoRA script.

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(**batch)
    output.loss.backward()
    torch.cuda.synchronize()
prof.export_chrome_trace("bc250-trace.json")
print(prof.key_averages().table(sort_by="self_device_time_total"))
```

The CPU-first initialization-order defect found during N2.12 is now fixed in
the selected .104 runtime. A CPU-only first capture can be followed by GPU
captures in the same process; CPU-only activity selection still omits GPU
kernel/copy records. No special first-capture workaround is needed. [Both-order
qualification and recovery](bc250_profiler_order_results/README.md).

Import `torch` in the script and warm up before selecting a short capture window.
Tracing is explicit; ordinary training does not automatically collect profiles.
It adds overhead and does not directly accelerate LoRA. Hardware counters,
simultaneous external profilers, compile/graph and long tracing are unqualified.
[Qualification, memory/schedule examples and rebuild/recovery](bc250_profiler_a09_results/README.md).

## Native primitive coverage (N2.16)

N2.17 additionally verifies FP32/FP64 `torch.sort(..., stable=True)` and
`torch.masked_select` on contiguous/strided inputs with CPU output, index and
finite-input backward references. Special-value sort fixtures check output and
indices only. Ordinary Torch calls and launcher defaults remain unchanged;
native C++ sorting/selection also passed explicit ordering/payload tests.
[N2.17 scope and evidence](bc250_sorting_n217_results/README.md).

The existing installed rocPRIM/hipCUB/rocThrust headers passed bounded FP32/FP64
reductions and ordinary/segmented scans, including guarded in-place operations,
custom argmax, empty segments and private streams. This extends C++ header
qualification; it introduces no Torch operator, launcher option, dispatch change
or measured training speedup. Broader Torch consumers and autograd paths remain
separate from these standalone tests. [Scope and results](bc250_primitives_n216_results/README.md).

## RNG replay and checkpoints (N2.15)

GPU RNG state is a CPU byte tensor: save `generator.get_state()` or
`torch.cuda.get_rng_state()` and load state tensors with `map_location="cpu"`
before restoring them. Exact continuation requires matching operation order,
shapes, types and execution path. CPU/GPU RNG algorithms need not produce the
same bitstream. Torch CUDA offsets must be multiples of4 and are per-thread
Philox offsets, not a count of output tensor elements.

Bounded strided/low-precision replay, high offsets, private streams and fresh
process continuation passed. Native LFSR113 C offsets already work; its stale
source comment was corrected, while the C++ wrapper still does not expose them.
No sampler/default/binary change or training speedup was introduced.
[Evidence and limits](bc250_rng_n215_results/README.md).

## FFT output and backward (N2.14)

Use ordinary `torch.fft` operations on CUDA tensors; no launcher flag or new
installation is needed. Bounded FP32/FP64 and complex64/complex128 tests cover
batched/strided transforms, odd real lengths, three normalizations, first
backward and guarded `out=`. Supply the original length when inverting an odd
real transform: `torch.fft.irfft(spectrum, n=17)`; omitting `n` assumes an even
length by API design. BF16 FFT promotion remains separate B01 work.

This batch adds qualification of the existing backend, with no dispatch change
or speed claim. Raw hipFFT/rocFFT callers have weaker input/stride-gap
preservation guarantees than Torch. See [scope and evidence](bc250_fft_n214_results/README.md).

## Sparse triangular solves (N2.13)

The selected .104 Torch runtime now marks missing or exactly zero diagonal
pivot rows as NaN for non-unit CSR/BSR `torch.triangular_solve(B, A, ...)`.
Selected FP32/FP64/complex64/complex128 layouts, indices and streams are qualified.
Nonsingular solves use the existing native backend plus one GPU diagonal check;
there is no speedup claim. Other rows of a singular result are unspecified.
Unit solves bypass the check; empty sparse matrices retain their old NaN behavior,
including unit mode. This is the existing deprecated sparse-compatible API;
general `torch.sparse.spsolve` and sparse solve autograd are separate gaps.
[Qualification, exact scope and recovery](bc250_sparse_n213_results/README.md).

## Corrected gated-delta and optional head256 attention (2026-10-18)

The .104 development runtime now uses reference-ordered **GPU** preparation/scan
for full FP16/BF16/autocast gated-delta. This closes the recorded BF16 parity gap:
all372 nonzero adapter gradients exactly match the reference in actual2B65/1024
fixtures. Native non-autocast FP32/direct preparation remains. This is not a full
FLA backend. The existing Qwen3.5 app hook remains enabled and now reports
“precision-safe GPU gated-delta active.” `BC250_GATED_DELTA=0` among the inner
`bc250-train --env` options selects the reference; shell assignment alone is not
inherited. Other projects must opt into the explicit API.

Math attention remains the default. For a **new output/run**, opt into head256:

```sh
~/finetune-n7/dlvenv/bin/python ~/finetune-n7/scripts/train_bc250.py \
  --attention-backend tiled --output ~/finetune-n7/checkpoints/new-tiled-run
```

BF16 D>128 training uses math forward plus native tiled backward. Saved state is
linear but forward temporary allocations remain quadratic. It reduced isolated
peak memory while running slower in the measured BF16 fixtures, so it is not an
automatic LoRA speed switch. A changed backend gets a distinct checkpoint
fingerprint. Default math preserves the earlier fingerprint schema; old
checkpoints are not rewritten. Exact fresh/resume continuation was checked
within each corrected backend, not across old buggy code/backend changes.
[Results, limitations and recovery](bc250_a06_a07_results/README.md).

## Sparse backward and measured cost (follow-up, updated 2026-10-16)

The current development runtime now supports first-order FP32/FP16/BF16 raw
structured-sparse gradients, including retained compression slots, dense operand,
bias and scalar/row alpha. Public FP16/BF16 semi-structured conversion preserves
the gradient to the original dense weight, projected onto the retained mask.
Matmul/linear backward uses dense GPU reconstruction where the transpose has no
valid row-sparse pack.38 CPU-reference tests and existing regressions pass.

This is explicit functionality, not an automatic training optimization. In12
small paired training fixtures, sparse forward/backward is1.92–3.36x slower than
dense; prepacked forward wins only for some smaller tested cases and packing adds
cost. Ordinary LoRA/dense behavior stays unchanged. Higher-order/compile/graph/
integer backward is unqualified. [Exact API scope, timings and working lint workflow](bc250_sparse_followup_results/README.md).

## Opt-in causal convolution and loss API (A05, updated 2026-10-16)

Scripts using the current `bc250-torch`/`bc250-train` can explicitly import
`causal_conv` and `chunked_causal_loss` from `torch.nn.bc250_training_ops`.
Native depthwise causal convolution now supports trainable input/weight/bias,
FP32/FP16/BF16, K1–4, SiLU, strided tensors and eager first-order autocast;
other valid convolution calls use ATen. The chunked causal loss retains original
logits and recomputes FP32 probabilities in backward. See the
[A05 API example, normalization rules and tests](bc250_training_a05_results/README.md).

This installs no global operator/model hooks. The existing LoRA trainer keeps
its earlier qualified helper until explicitly migrated. Compile/graphs, higher-
order gradients, dynamic scaler overflow and large-model speedups are unqualified.
The shared frozen consumer bundle and fallback remain unchanged.

## Structured sparse Torch operations (A04, updated 2026-10-16)

The current development `bc250-torch`/`bc250-train` runtime supplies native scalar
hipSPARSELt compression and sparse matmul. `torch.backends.cusparselt` is the
shared Python namespace on HIP too; `is_available()` now reflects compiled
support and `version()` returns209 (installed hipSPARSELt0.2.9).

Explicit `_cslt_compress`/`_cslt_sparse_mm` calls support same-dtype FP32, FP16,
BF16 and INT8 outputs. Inputs must already have valid structured sparsity:
FP32 uses1:2, the other types2:4; compression does not magnitude-prune a dense
matrix. FP16/BF16 public `torch.sparse.to_sparse_semi_structured` plus `torch.mm`
were also checked. GPU scalar or contiguous FP32 per-row vector alpha and FP32/
matching-floating-type bias are supported. Widened outputs/split-K are refused.
See [tested contracts and limits](bc250_torch_a04_results/README.md).

This is explicit sparse execution on the GPU; ordinary dense operators continue
to use their existing backend. The follow-up section above supersedes the original sparse-gradient gap.
Compilation/graphs and universal sparse speedups remain unqualified. Other prefixes/venvs and the frozen fallback are unchanged.

## Convolution correction (updated 2026-10-16)

The current development prefix now includes the A01 ROCr clock-epoch and MIOpen
dilated-workspace corrections. Programs launched through `bc250-torch` or
`bc250-train` use them automatically; no training-script flag is required.
Fifteen direct API and30 Torch convolution output/gradient configurations passed.
Other installations/venvs are outside this deployment, and these bounded checks
do not qualify every convolution or establish a speedup.

A02 also corrects the bundled legacy CK target rejection and BF16 reduction
conversions/scaling. The installed runtime passes54 direct numerical reduction
cases, three unsupported-compute refusal checks and12 FP32/FP16 LSTM/GRU/RNN
output/gradient cases. This applies automatically to callers loading this MIOpen;
BF16 RNN remains unsupported. A03 now supplies the bounded system database seed
and capability-aware optional-plugin probing described below. See [A02 results and limits](bc250_miopen_a02_results/README.md)
and [A01 convolution limits](bc250_miopen_a01_results/README.md).

## MIOpen databases and optional plugin (updated 2026-10-16)

A03 installs a genuine gfx1013_20 seed database in the current development prefix:
16 convolution keys/58 checked solutions, one searched performance configuration
and41 compiled kernel records. `bc250-torch`/`bc250-train` use it automatically.
This covers the listed small FP32/FP16/BF16 NCHW fixtures; other shapes retain
normal search/compilation and user-cache updates. It is not global tuning.

The default loader now skips the modern CK plugin when build/device capability
disables it, so its expected absence no longer produces a warning. This adds no
CK grouped-convolution kernels. An explicit `MIOPEN_CK_LIB_PATH` still diagnoses
missing or incompatible experimental plugins. Source/payload scope and hashes:
[A03 report](bc250_miopen_a03_results/README.md). Other ROCm prefixes and the
frozen fallback remain unchanged. Missing causal-conv/FLA packages are separate
A05/A06 work; this does not change their warnings.

## Storage and software fallback (updated 2026-10-16)

The active development runtime is unchanged by the storage cleanup. Both
`/opt/bc250-stack/previous` and `before-dev-current` now select the retained,
independently frozen `releases/10.2-gfx1013-20261013-4b000d542` fallback. Its
consumer bundle still supplies the normal training launcher. Active and fallback
passed short GPU reference checks after cleanup; the kernel boot fallback is
unchanged.

The superseded initial20261012 release and standalone priority2-20261014
candidate retain metadata/validation only; they cannot be launched. Old synthetic
validation checkpoint binaries were also retired and must be regenerated by test
helpers if needed. Your training models, datasets and real checkpoints remain.
See the [cleanup journal and scope](bc250_storage_cleanup_20261016/README.md).

## Qwen3.5 training on .104 (updated 2026-10-16)

For stack-wide availability, reproduced defects, missing optional packages and
the ordered work queue, see the [Torch support matrix](bc250_torch_support_matrix.md).
The Qwen3.5 helpers below are application scoped; they do not replace ordinary
Torch convolution or loss globally. A01 corrects the reproduced general-convolution
timing/workspace failures; A02 corrects the legacy CK reduction failure and
qualifies the recorded FP32/FP16 recurrent output/backward fixtures.

The current `finetune-n7/scripts/train_bc250.py` launcher uses `replica/train.py`
with the project's Transformers5.19.0 overlay. It now enables native gfx1013
frozen causal convolution and chunk64 FP32 loss by default. The source hash is
printed at startup. These operations avoid MIOpen search for this convolution
and the loss's full FP32 logits copy; full BF16 logits still exist. No package
installation or checkpoint migration is needed. Resume also accepts PEFT's
equivalent shortened target names while requiring the exact resolved modules,
rank and alpha. Bounded fresh/resume continuation passed.

Gated-delta now uses the corrected GPU operation described above; complete native
FLA fusion is absent. A01 corrects reproduced MIOpen timing/workspace faults
within its recorded scope, not all operations. Qwen3.5 keeps math SDPA by default;
head256 tiled/hybrid is now an explicit option as described above.
See [usage and opt-out](../../build_tools/bc250/qwen35_training_usage.md), also
installed as `/home/nik/finetune-n7/BC250_TRAINING_OPS.md`, and the
[qualification report](bc250_qwen35_training_results/README.md).

## Attention selection on .104 (updated 2026-10-15)

The incorrect `1Torch was not compiled with memory efficient attention` warning
is fixed in the current development runtime. The installed `finetune-n7`
`train_lora.py` and `chat.py` also use the supported `sdpa_kernel` API; training
backward stays inside the context for checkpoint recomputation. These corrections
apply by default. They do not require enabling the new tiled backend.

**Math attention remains the default.** The memory-saving GPU tiled backend is
opt-in. For those two installed scripts, append `--attention-backend tiled` to
your existing command, **after the script path**, keeping all model/data/output
arguments. For example, to start chat with the optional backend:

```sh
bc250-train /home/nik/finetune-n7/scripts/chat.py \
    --model /home/nik/finetune-n7/models/Qwen3-1.7B-Base \
    --attention-backend tiled
```

Use `--attention-backend math` or omit the flag to retain the default. The flag
belongs to these application scripts, not the `bc250-torch`/`bc250-train` launcher;
it does not automatically work in arbitrary third-party scripts.

For your own Transformers application, explicitly register the backend before
loading the model:

```python
from torch.nn.attention._gfx1013 import register_transformers_attention

register_transformers_attention()
model = AutoModelForCausalLM.from_pretrained(
    model_path, attn_implementation="bc250_tiled", dtype=torch.bfloat16
)
```

This explicit backend supports eager first-order FP32/FP16/BF16 attention, grouped heads,
head width up to256 and sequence length up to4096, with the documented mask
contracts. Attention dropout, mask gradients, higher-order gradients and compiled/
graph operation are unqualified or unsupported. Ordinary functional SDPA keeps its
math fallback. Tiled attention supports CUDA autocast. Causal head128 kernel fixtures improved
with tile skipping, but this is not a whole-model speed guarantee. Bounded real
Qwen3-1.7B LoRA references at512/1024tokens passed; full training/convergence remains
unqualified, and the whole-model1024-token peak stayed unchanged in our fixture.
See the [training follow-up](bc250_training_expansion_results/README.md) and the
[attention report](bc250_attention_results/README.md) for references, measurements,
exact contracts and rollback information.

## Resume state and crash diagnostics (updated 2026-10-15)

The installed `train_lora.py --resume CHECKPOINT` keeps adapters trainable and
restores GPU RNG from the required CPU byte tensor. New checkpoints also save
scheduler, Python RNG and max-steps shuffle/remaining-index state. Existing
old-format checkpoints remain usable, with warnings for unavailable exact replay
state. Retain matching schedule arguments; restoring a completed cosine schedule
does not create a new learning-rate plan. Bounded new/legacy fixture resumes and
separate fresh-process AMP/optimizer/RNG continuation tests passed; this is not
universal checkpoint/model compatibility qualification.

The launcher defaults `PYTHONFAULTHANDLER=1` so future fatal Python faults produce
tracebacks in your logs. Normal core-dump policy stays unchanged. To opt out
explicitly, use launcher option `--env PYTHONFAULTHANDLER=` before the script path.
The value `0` still enables Python's handler because it is a nonempty string.
The historical CPU segfault remains unresolved; see the follow-up report above.

## Paths and persistence

Host paths retain their names inside the container. Your home directory, current
working directory, and an external script's parent directory are mounted
read-write automatically. The working directory remains the directory where you
called the launcher; it does not change to the script directory.

| Scenario                                    | What happens                                                                                              |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `torch.save(..., "checkpoints/model.pt")`   | Writes under the caller's host working directory; survives container exit                                 |
| `/home/nik/models/model.pt`                 | Uses the same absolute path on the host and inside Python                                                 |
| `Path.home()` and `os.path.expanduser("~")` | Resolve to `/home/nik`                                                                                    |
| Script outside the current directory        | Its directory is mounted at its original path; sibling imports/files remain accessible                    |
| `Path(__file__).parent / "config.json"`     | Reads the host file next to the script                                                                    |
| Host edits a mounted file between runs      | The next container sees the host edit                                                                     |
| Container edits a mounted file              | The host sees the edit immediately; ordinary file locking/atomic replacement rules still apply            |
| Files named with spaces or Unicode          | Supported; quote paths in the shell                                                                       |
| Symlink into another directory              | The target must also be mounted; home and cwd are automatic, external targets need `--mount`/`--mount-ro` |
| `/tmp` and unmounted container directories  | Temporary; use an explicit host mount if their contents must survive                                      |
| New Python process/container                | Mounted checkpoints, downloads and extra packages remain; Python in-memory state starts fresh             |

Creating a directory inside a mounted directory creates it on the host too. An
explicit mount source must already exist. The launcher does not guess dataset
paths from script arguments or scan your code for paths.

For files on an external disk or outside the automatic mounts:

```bash
mkdir -p /mnt/models/checkpoints
bc250-torch --mount-ro /mnt/datasets --mount /mnt/models train.py \
  --dataset /mnt/datasets/train.json --output /mnt/models/checkpoints
```

Both explicit options support an existing file or directory and preserve its
absolute path. `--mount-ro` prevents writes through that mount. Automatic cwd/script mounts
inside a read-only parent inherit its mode. An explicit `--mount` can make a
specific output subdirectory writable. Filesystem
permissions still apply; the container gets the user's supplementary groups.
Existing filesystem access on the host does not automatically mount every disk.
A symlink to `/mnt/datasets` also needs that target mounted.

## Launcher options and Python arguments

Launcher options precede Python/script arguments. Once Python arguments begin,
remaining options belong to Python or your script. An optional `--` ends launcher
option parsing.

```bash
bc250-torch --launcher-help
bc250-torch -u ~/projects/training/train.py --epochs 10
bc250-torch --mount-ro /mnt/data -- -m my_project.train --data /mnt/data
bc250-torch -c 'from pathlib import Path; print(Path.cwd(), Path.home())'
printf 'print("stdin works")\n' | bc250-torch -
bc250-torch --shell
bc250-torch --shell -c 'ls "$HOME/models"'
```

Shell mode keeps the validated runtime PATH and skips host Bash startup files.

`-h` and `--help` are forwarded to Python; use `--launcher-help` for launcher
help. The launcher returns Python's exit status. Doas may request your password
before creating the container. Regular host authentication requirements also
apply to piped or background invocations.

## Checkpoint save and resume

Save model/optimizer state rather than keeping it only in the container's memory:

```python
from pathlib import Path
import torch

checkpoint = Path("checkpoints/latest.pt")
checkpoint.parent.mkdir(parents=True, exist_ok=True)
temporary = checkpoint.with_suffix(".tmp")
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": torch.get_rng_state(),
        "gpu_rng": torch.cuda.get_rng_state(),
    },
    temporary,
)
temporary.replace(checkpoint)
```

The temporary file is in the same directory as the final checkpoint, so readers
see the old or new file after replacement. This handles ordinary save completion;
it does not promise power-loss durability. For concurrent writers use unique
paths or an application-level lock.

Resume in another `bc250-torch` invocation:

```python
state = torch.load("checkpoints/latest.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
torch.set_rng_state(state["rng"])
torch.cuda.set_rng_state(state["gpu_rng"])
step = state["step"]
```

Our gate saves GPU model/Adam state, restores it in a new container and continues
training. It also verifies CPU loading with `mmap=True`. Scheduler, AMP scaler,
sampler and application-specific state should be saved if your training needs
them; the launcher cannot reconstruct state that your program did not save.

The host can copy, back up or transfer checkpoints with normal tools:

```bash
ls -lh ~/projects/training/checkpoints
scp ~/projects/training/checkpoints/latest.pt another-host:/path/to/backups/
```

Python checkpoints need compatible loading code; bind mounts do not change the
checkpoint format. Weight/checkpoint/model-library compatibility is separate
from the file transport validated here.

## Downloads, model caches, credentials and network

The launcher uses host networking, so code can download files and reach services
on the host's `127.0.0.1`. Files downloaded under your home/project directory
persist. Default home-based Hugging Face, Torch Hub and pip cache locations are
on the host; no additional mount is needed for those locations.

Common `HF_*`, `HUGGINGFACE_*`, `WANDB_*`, `MLFLOW_*`, HTTP proxy settings,
`TOKENIZERS_PARALLELISM`, thread limits, device-visibility variables,
`XDG_CACHE_HOME` and `TORCH_HOME` are carried across `doas`. Inherited names are
listed in `INHERITED_NAMES`/`INHERITED_PREFIXES` in `bc250_torch.py`. Other values
can be passed explicitly:

```bash
export MY_TRAINING_SETTING='value with spaces'
bc250-torch --env MY_TRAINING_SETTING train.py
bc250-torch --env HF_HOME=/mnt/models/hf-cache --mount /mnt/models train.py
bc250-torch --env-file ~/training.env train.py
```

Environment files contain literal `NAME=VALUE` lines, with blank lines and `#`
comments allowed. They do not evaluate shell expressions or remove shell quotes.
For example, write `MY_SETTING=value with spaces` directly. Values must be single
lines. Export tokens on the host or use a permission-restricted environment file;
`--env NAME` reads the caller's value without putting it in the Podman argument
list. The elevation handoff and Podman env file use private temporary files and
are cleaned up after the command; their contents are not logged.

Values selecting the validated runtime, including `HOME`, `PATH`, ROCm library
paths, SDMA and compiler cache locations, are controlled by the launcher. Their
replacement is rejected. Other library-specific model-cache locations can be
set normally, but external paths still need a corresponding mount.

The download gate uses a real HTTP transfer from a temporary host loopback server,
then reads and hashes the downloaded file in a subsequent container. It does not
establish connectivity or credentials for every external provider.

## DataLoader workers and shared memory

The default `/dev/shm` limit is **1 GiB**. This is a limit rather than an up-front
allocation. It avoids Podman's usual 64 MiB default for training workers; the
option is documented in the [Podman run reference](https://docs.podman.io/en/latest/markdown/podman-run.1.html#shm-size-number-unit).

```bash
bc250-torch --shm-size 2g train.py
```

The board shares physical memory between CPU and GPU. Larger limits do not add
physical RAM; choose batch size, worker count and prefetching to fit the workload.
The gate touches 80 MiB of shared memory and checks a two-worker DataLoader with
`multiprocessing_context="spawn"`, persistent workers, pinned memory and exact
CPU-reference batches over two iterations. Keep the normal Python
`if __name__ == "__main__":` entry-point guard when using spawn.

## Extra Python packages

The validated image contains an immutable Python environment. Install additional
packages into a mounted directory if they must survive container exit:

```bash
mkdir -p ~/bc250-python-packages
bc250-torch -m pip install --target ~/bc250-python-packages --no-deps PACKAGE
bc250-torch --env PYTHONPATH="$HOME/bc250-python-packages" train.py
```

This workflow was tested with an offline local pure-Python package, imported in a
new container. `--no-deps` keeps pip from replacing the validated Torch/Triton and
numerical dependencies; install required compatible dependencies explicitly.
Do not use Alpine-built binary Python extensions in the glibc container. An
extra package can shadow an installed package through `PYTHONPATH`, so its own
compatibility still needs checking. Packages installed only into a disposable
container layer do not survive exit.

## Interruption and detached jobs

Terminal Ctrl-C is passed to the foreground runtime. The host integration gate
also checks SIGINT and root-issued SIGTERM with an application handler that writes
a final host-visible file before exiting. A graceful checkpoint on interruption
requires a handler in your training program; abrupt termination cannot complete a
save automatically.

Rootful Podman is started through `doas`. For administrative termination of a
detached rootful runtime use root permissions; killing just an unprivileged
wrapper is not a reliable way to stop the container. Authenticate normally when
starting detached work and use persistent host paths for its logs and outputs.
Distributed training and arbitrary third-party launchers are separate runtime
qualifications; this file workflow does not establish RCCL/multi-GPU support.

## Verification and recovery

Thirteen host integration scenarios passed on 2026-10-13. The actual `nik`
elevation handoff also passed environment and GPU-checkpoint checks, piped
stdin and interactive shell mode. The user matrix script passed after installation.
Read-only parent/cwd inheritance and an explicit writable output exception were
also verified on the host.

The host integration runner is `build_tools/bc250/validate_launcher_on_host.py`.
It can test a staged launcher with `--launcher PATH` before installation. Evidence
is stored under the `.104` workspace's `logs/launcher-io/`; sample checkpoints
are under `launcher-io-validation-*/training project/checkpoints with spaces/`.

The previous installed launcher is preserved as
`recovery/bc250-torch-before-training-io-20261013` in the workspace.
Launcher source revision/hash and evidence are recorded separately in
`/opt/bc250-stack/launcher.json`. Existing release-home compilation caches were
merged into the host home cache without overwriting existing files.
The runtime's `/opt/bc250-stack/current` and `/opt/bc250-stack/previous` symlinks
are independent of the launcher update. See [bc250_port.md](bc250_port.md) for
runtime recovery and [bc250_full_support_roadmap.md](bc250_full_support_roadmap.md)
for remaining ROCm work.

Real pretrained full fine-tuning, FP16/BF16 LoRA and fresh-container training
resume are now qualified for the small recorded workload. Use the isolated
consumer environment and examples in [bc250_training_memory.md](bc250_training_memory.md).
