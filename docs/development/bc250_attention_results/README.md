# BC-250 scalar tiled attention — 2026-10-15

Implemented in qualified patched Torch `bda0831ba49ef9d0ba2968a800642e5abaec5dd5`. Native gfx1013 execution on .104;
no architecture override, BIOS/kernel/governor changes or .105 writes. Mathematical
SDPA remains the default. This is an opt-in eager, first-order-autograd path,
not an enabled built-in FlashAttention/memory-efficient dispatcher backend.

## Changes

- `torch/nn/attention/_gfx1013.py`: three Triton scalar kernels use a16-key/query
  tile, online maximum/normalizer, FP32 accumulation and backward recomputation.
  They never allocate the full attention score/probability matrix and use no
  `tl.dot`/matrix instructions. Each KV row owns grouped-head gradients, without
  atomics or replicated KV tensors. FP32 output+row logsumexp are saved for backward.
- Explicit Transformers registration includes **both attention and mask registries**;
  omission of the mask registration could silently lose padding/packed-position
  semantics. Registration never occurs on ordinary Torch import.
- C++ probe bug fixed: `TORCH_WARN_ONCE(!debug, text)` printed `1Torch` during normal
  dispatch. Match the flash probe's `if (debug) TORCH_WARN(text)` behavior. Quiet
  normal selection is separate from actual efficient-attention implementation;
  forced built-in efficient selection still fails and explicit diagnostics remain.
- User `train_lora.py`/`chat.py` migrated to `nn.attention.sdpa_kernel(MATH)`.
  Training backward stays within that context, including checkpoint recomputation.
  Broad warning filters removed. New `--attention-backend tiled` explicitly selects
  the prototype; default `math` retains the previous backend. Reviewed patch:
  `build_tools/bc250/patches/finetune-n7-sdpa-context.patch`.

## Bounded evidence

- `sdpa-tiled-candidate-r5`:18 configurations, each output and Q/K/V gradients checked
  against independent CPU float64 manual attention and GPU math SDPA. FP32/FP16/
  BF16, GQA, noncontiguous inputs, dimensions7/16/32/33/64/128, rectangular causal
  attention, boolean/additive masks, fully masked rows, an empty initial key tile,
  negative scale. Maximum CPU-relative L2 across the18 cases:2.58e-7/2.32e-4/1.86e-3
  for FP32/FP16/BF16. Elementwise tolerances and results retained in source/JSON.
  Also empty lengths, five option rejections, activation checkpointing, and padded
  one-layer FP32 Llama output/parameter gradients versus CPU float64 math SDPA.
- `sdpa-tiled-installed-integration-r4`: seven groups pass. Quiet normal/default
  selection plus retained explicit diagnostics/failure, scale0 and10 output/gradients,
  nondefault GPU stream, BF16 causal65-token Hq16/Hkv8/head128 CPU references (the
  user's Qwen3-1.7B head geometry; relative L2 <=1.68e-3), and a **small random**
  one-layer Qwen3 BF16 LoRA model with two AdamW updates/checkpointed backward.
  CPU float64 model logits relative L2 <=0.00431, LoRA gradient relative L2 <=0.00860;
  cached decode matches full-prefix evaluation. Deployed caller AST/CLI checks pass.
  This does not qualify full Qwen3-1.7B training or existing checkpoint resume.
- `sdpa-torch-regression`: unchanged eager19-reference bmm/FFT/Cholesky/sparse/Adam
  regression passes on rebuilt loaded Torch. All final batches exit0, guards ACKed,
  kernel logs byte-identical before/after; no warning filters used in those tests.

Measured extra **PyTorch allocated** peak bytes for forward+backward, contiguous
FP32 B1/H2/D32, no mask, after kernel warmup (not total device/process memory):

| Length | Tiled MiB | Math MiB | Tiled ms | Math ms |
| --- | ---: | ---: | ---: | ---: |
|512|0.630|8.376|1.22|1.03|
|1024|1.259|32.751|2.54|2.36|
|2048|2.517|129.501|8.29|6.58|

These are one-event illustrative timings, not a performance qualification. Tiled
was slower in these fixtures. The measurements demonstrate linear auxiliary
allocation for these shapes; those larger fixtures were memory/timing tests,
not additional numerical CPU references. Caller-provided dense masks can themselves
consume quadratic storage; this backend does not remove that input storage.

## Usage

```python
from torch.nn.attention._gfx1013 import scaled_dot_product_attention
output = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
output.sum().backward()
```

For the installed user scripts, retain existing arguments and append
`--attention-backend tiled`, for example:

```sh
bc250-train /home/nik/finetune-n7/scripts/train_lora.py <existing arguments> --attention-backend tiled
bc250-train /home/nik/finetune-n7/scripts/chat.py <existing arguments> --attention-backend tiled
```

For other Transformers callers, explicitly call `register_transformers_attention()`
from the same module, then select `attn_implementation="bc250_tiled"`.
Ordinary `F.scaled_dot_product_attention` continues using existing math fallback.

## Contracts and remaining scope

Q/K/V share GPU, dtype and batch, `[B,H,L,D]`, equal key/value shape and head width;
FP32/FP16/BF16, D1..128, L0..4096, tensor elements<2^31. Grouped query heads must be
an integer multiple of KV heads and explicitly enabled. Masks broadcast to
`[B,Hq,Lq,Lk]`, boolean or compatible float, no mask gradients; finite Q/K/V and
finite/-inf mask values. Noncontiguous Q/K/V are packed without losing gradients.
No dropout, higher-order gradients, compiled/graph qualification, softcap or
sliding-window integration. Explicit mask+causal is rejected. Do not advertise
universal attention API compatibility or a speedup.

Next attention work: longer head128 sequence references and representative full-model
training/memory/time comparisons before any default integration; then targeted
optimization and optional contracts. K6/N2.1 remain the independent kernel/library
roadmap. Python CPU segfault remains unresolved and separate from repaired VM-heap
corruption and kernel VMM/SVM fixes.

## Rebuild, deployment, recovery

Host WS=`/home/nik/bc250work/rocm-10.2-gfx1013`. Compile used existing runtime/build
image5df0a58392de..., native gfx1013,12jobs/20GiB ccache. Targeted regeneration and
build compiled SDP-utils plus regenerated HIPHooks, then linked libtorch_hip;
loaded Torch path must receive the resulting `build/lib/libtorch_hip.so`.
`build_torch_sdpa.sh` and `deploy_sdpa_probe_fix_on_host.py` preserve that procedure.

Every mutation/test used `WS/helpers/update_dev_prefix.py -- COMMAND` for the
exclusive development-prefix lock. Validation runner constructs Podman directly,
never nests a shared launcher lock. Fresh600s recovery guards were ACKed; no reboot.
`validate_tiled_sdpa_on_host.py --label UNIQUE` tests a staged candidate;
`--installed --integration` tests installed callers/models;
`--installed --regression` runs the retained19-reference gate. Logs are
`WS/logs/LABEL/`; before/after raw kernel logs remain there.

Retained rollback files: `WS/recovery/sdpa-probe-before/{sdp_utils.cpp,libtorch_hip.so,identity.json}`
and `WS/recovery/sdpa-callers-before/{train_lora.py,chat.py,identity.json}`.
Review current hashes before restoring under the same exclusive lock; do not
replace subsequent user edits. Math default requires no rollback to stop using
the prototype. Original images/releases, builds/cache and checkpoints are retained.

Earlier failed runs are preserved, not promoted to passes: first consumer mount
path mismatch; r2 final warning check while old library loaded; r3 CPU eager
reference produced NaNs on fully masked rows (CPU math SDPA used subsequently for
its defined zero-row semantics); r4 still loaded old `torch/lib` until the built
library was copied from `build/lib`. First installed diagnostic harness captured
only Python warnings, not c10 stderr; second had a loss dtype comparison mismatch.
Final tests retain the original numerical tolerances and include mask registration.
