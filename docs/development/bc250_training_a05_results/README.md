# A05: reusable opt-in training API — 2026-10-16

Installed .104 source: Torch `6181d00b5beb8cc9fd6800da7a19c8abf756b30d`,
TheRock helpers `869463155599a1c836bef10fb6089cbaa511058c`.
[Deployment/source/hashes](deployment.json), [69 installed tests](api-installed.json),
[core regression](core-installed.json).

## API and routing

Scripts launched with `bc250-torch` or `bc250-train` can explicitly import:

```python
from torch.nn.bc250_training_ops import causal_conv, chunked_causal_loss

# x[B,C,L], weight[C,K], bias[C] or None; optional activation="silu"/"swish".
y = causal_conv(x, weight, bias, activation="silu")
# logits[..., vocabulary], int64 labels matching the token dimensions.
loss = chunked_causal_loss(logits, labels, chunk_tokens=64)
loss.backward()
```

No model-name/Transformers-version dependency or global operator/model patch is
installed. Existing `finetune-n7/replica/bc250_ops.py` and trainer remain unchanged
on the previously qualified frozen-convolution implementation. Existing training
scripts keep their behavior until they call the new API. Do not blindly assign
this loss to an arbitrary model's loss hook: unsupported keyword semantics raise
rather than being silently ignored.

Native convolution selects actual gfx1013, matching FP32/FP16/BF16 tensors,
nonempty depthwise K1–4, C<=65535, B*L<=1048576 and total elements<2^31.
`supports_causal_conv(x,w,b,activation)` reports the direct native predicate;
`causal_conv` additionally handles CUDA autocast, casting FP32 inputs/parameters
to the configured FP16/BF16 compute dtype. Native forward retains the prior
round-before-SiLU convention and input-gradient kernel. New deterministic FP32
partial/reduction kernels compute weight and bias gradients without floating
atomics. Input-gradient work is skipped when unrequested. Strided inputs,
weights/bias and custom streams are supported within tested bounds. Other valid
convolution calls use normal ATen; unsupported activations or malformed matrix
shapes raise. This is eager first-order only, without an incremental decode cache.

Chunked loss retains the prior recomputed-FP32-probability algorithm. It accepts
contiguous FP32/FP16/BF16 logits, int64 labels, ignore_index=-100 and chunk1–256;
normal labels are causally shifted, or explicit shift_labels bypass shifting.
Default mean uses the nonignored target count; explicit positive finite scalar
num_items_in_batch normalizes a sum. All-ignored mean returns NaN/zero gradients,
matching ATen. Invalid vocabulary/label/count/chunk/unknown-keyword calls fail
before GPU index assertions. CPU execution is available too; numerical tests
below primarily qualify native gfx1013. Full original logits and their gradient
remain resident. There is no fused linear projection/loss or FLA implementation.

## Executed evidence

Final candidate and installed suite:69 device-generic Torch TestCase tests:

- 48 native convolution cases: FP32/FP16/BF16, K1–4, B1–2/C3–9/L1–1031,
  SiLU/no activation, contiguous/strided input, strided weights/bias. CPU double
  references with explicit low-precision rounding for output and dx/dw/db.
  Both single and multiple partial-reduction tiles; custom streams and unchanged
  source storage checked.
- Three no-bias tests, each with/without SiLU and trainable x/weight gradients.
- Four AMP tests: FP16/BF16 autocast from FP32 parameters, with/without
  non-reentrant checkpoint recomputation; CPU rounded references for all
  gradients, fixed8x scaling/unscaling and an SGD update. This does not establish
  dynamic GradScaler overflow/growth behavior.
- Twelve loss cases: three dtypes, mean/count/pre-shifted/all-ignored variants;
  CPU loss/gradient references and scaled upstream derivative.
- One dispatch/guard group: six input-only/weight-only/bias-only native K1 and
  ATen-fallback K5 gradient checks, CPU fallback and eight refusal contracts.
- One synthetic trainable convolution + projection + chunked-loss model: CPU
  loss/all-parameter gradients, checkpoint recomputation, AdamW update,
  serialized parameters/optimizer/GPU RNG and exact save/resume continuation.
  It is a tiny model gate, not an actual Qwen convergence/training qualification.

Installed regressions pass seven core reference groups and the20 A04 sparse
cases/120 small sparse matmuls. Six total GPU guards (candidate65, candidate-r2
69, final69, installed API/core/sparse) all ACK within600s; dmesg identical
SHA53f83f7bbee6c060e3153e7ae95a4ff069bf9627113591c39cdddb9ab5f4b973.
No GPU correctness failures or tolerance relaxation.30 protected hashes unchanged,
including all active Torch DSOs, ROCm/DB/kpack, old Torch, frozen fallback,
consumer training.env and original application files. Final module SHA
06da955bc31f2d458a4929b062ea0319e801e9bc71bcadee7be318d87ca39a23.

## Deployment, provenance and limits

Exclusive runtime lock spans module staging, installed gates and rollback.
One Python source file was atomically added to the existing active
`WS/pytorch-a04/torch/nn/`; no C++ build or DSO changes. Native kernels JIT through
the existing patched Triton/cache/compiler. Shared/frozen python-consumer bundle
was deliberately preserved; it is shared by current and fallback. Recovery
metadata/stamp: `WS/recovery/training-a05-before`. Failed gates would remove only
the newly added module and restore metadata. WS is
`/home/nik/bc250work/rocm-10.2-gfx1013`.

Raw logs: `WS/logs/training-a05-*`, local ignored
`kernel/artifacts/training-a05/`; SOURCE_TRAINING_A05.json in WS/current. Full
local Torch Git, incremental patch0004 and verified Git bundle based on8cdfcc8
are retained. Historical three-file SDPA patch-bundle gap remains separate.
Python syntax/Git whitespace checks pass; spin/lintrunner attempts remain blocked
by missing offline lint initialization, so no lint pass is claimed.

No global/default speedup, large-model memory/convergence, higher-order gradients,
compile/export/graph capture, determinism across compiler/device versions or
sustained stability qualification. The prior frozen helper's real Qwen/model/CLI
results are historical evidence, not tests of this new module. No active agent
jobs remain. Kernel/boot/BIOS/.105/user files/default dense BLAS/SDMA0/heap-off/math
SDPA unchanged. Historical Python CPU segfault remains unresolved and distinct
from fixed VM-heap/CSR corruption.

Exact next **A06**: inspect existing Qwen3.5 gated-delta/FLA reference and recorded
model shapes, profile bounded forward/backward substages, then implement one
hardware-feasible scalar/wave32 substage with independent output and Q/K/V/gate
gradient references before deployment. A06–A09 remain before N2.7; K6 completed.
