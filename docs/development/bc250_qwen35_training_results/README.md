# Qwen3.5 training operations — 2026-10-16

> **Fixture correction (2026-10-18):** `from_pretrained()` leaves actual models
> in eval mode. Earlier actual-model tests that only enabled checkpointing did
> not prove checkpoint recomputation. The shared fixture now calls `.train()`;
> [A06/A07](../bc250_a06_a07_results/README.md) asserts training/checkpoint flags
> and twelve actual head256 calls. Earlier real CLI training/resume remains distinct.

Implemented and enabled two application-scoped native gfx1013 training helpers in
`/home/nik/finetune-n7/replica/train.py` on .104. Existing `scripts/train_bc250.py`
uses them by default through `replica/bc250_ops.py`; other projects are unchanged.
Implementation revision: `690042b55f33b4a6fd67c4c48d54ff8d7313b64a`.
[Usage/opt-out](../../../build_tools/bc250/qwen35_training_usage.md),
[reviewed caller patch](../../../build_tools/bc250/patches/qwen35-training-ops.patch).

## Findings and changes

The retained user log is `~/finetune-n7/logs/qwen35-train-20261015-220758-837706.log`.
It continued to step251 and saved checkpoint250. Its allocator warnings describe
failed allocation attempts, not proof of a fatal OOM: qualified Torch
`c10/cuda/CUDACachingAllocator.cpp:3933` logs a failed hipMalloc, clears the error,
and permits cache release/retry before the final throwing path. An allocation of
1,017,118,720 bytes is exactly1024*248320*4, matching Qwen3.5's full FP32 logits
cast in Transformers5.19.0 `ForCausalLMLoss`. Fragmentation is not established.

1. Frozen causal depthwise convolution K<=4 now uses scalar Triton forward and
   input-gradient kernels, FP32 accumulation and explicit rounding before SiLU.
   FP32/FP16/BF16, bias/no bias and strided inputs are bounded-tested. Trainable
   weights/bias or unsupported cases retain the original function. No matrix
   instructions or architecture override. Cached decode convolution is unchanged.
2. Chunk64 causal loss computes FP32 cross entropy per token chunk and recomputes
   FP32 probabilities in backward. Mean/explicit item-count normalization, shifted
   labels and -100 masking remain. Full low-precision logits and their gradient
   still exist. Eager first-order only; this is not fused linear cross entropy.
3. Fresh/resume testing reproduced an independent application defect: PEFT0.18.1
   minimizes a large full-path target list to equivalent suffixes when injecting
   adapters. The old string-set comparison then rejected its own saved adapter.
   Resume now resolves the saved config with pinned PEFT's own matching helper and
   requires the exact requested module set, rank and alpha. Missing/extra targets
   and wrong rank/alpha are still refused. Checkpoint contents/fingerprints and
   model/data/tokenizer/schedule comparisons are unchanged.

## Executed gates

All listed successful runs used the exclusive development-prefix lock, pinned
rootful image, native gfx1013 and fresh600s recovery guards. They exit0, guards
ACKed and kernel logs byte-identical. Source identities are embedded in each JSON;
initial operator/memory/tiny runs precede formatting/type hints/version pin only.
The final operator, actual-model and installed-CLI gates use deployed SHA below.

| Label / evidence directory | Bounded result |
| --- | --- |
| `qwen35-ops-first` | Initial50 checks passed; retained separately |
| `qwen35-ops-final` |51 checks:36 CPU-float64 output/dx conv cases across three dtypes, actual BF16 B1/C6144/L1024/K4 geometry,12 loss/gradient cases, trainable-weight refusal, nondefault-stream/padded/strided-weight/bias/input guards |
| `qwen35-loss-memory` | Same seeded BF16 B1/L1024/vocab248320 logits: reference versus chunk64 loss, sampled gradient and total gradient norm agreement |
| `qwen35-model-tiny` |4 LoRA reference pairs, FP32/BF16 with/without non-reentrant checkpointing;30 adapter gradients each; max relativeL2~9.12e-7 FP32 and0 BF16 |
| `qwen35-model-actual` | Downloaded text-only Qwen3.5-2B, BF16/rank16/alpha32/all decoder Linear targets:65-token reference loss equal,372 adapter gradients max relativeL2~0.01121; separate synthetic1024-token checkpointed backward and AdamW adapter update passed |
| `qwen35-cli-resume-fixed` | Actual deployed trainer, synthetic42-token records, rank4:2 fresh updates/evaluations, resume checkpoint1 to step2 in a new directory. All372 adapter tensors and full optimizer/scheduler/RNG/progress state exactly equal uninterrupted result |
| `qwen35-resume-guards` |4 deliberately invalid synthetic adapter configs correctly exit2 before writing output: missing/extra target, wrong rank, wrong alpha |

Convolution small-case max output/dx relativeL2: FP32~9.25e-8/1.13e-7,
FP16~0/2.33e-4, BF16~0/2.85e-3. Actual-channel BF16 output/dx~1.49e-8/1.66e-3.
Fixed tolerances are in the helper; no tolerance was changed to make a failure pass.
Model references use an independent ATen GPU tap-sum convolution plus original
Transformers loss; convolution itself has independent CPU-float64 references.
All actual model tensors/operations stay GPU; CPU reference tests/state comparison
are explicitly test oracles. No user conversations, datasets or existing adapters
were used or modified. Synthetic checkpoints are only in these test log directories.

Isolated loss extra peak allocation: reference3,051,357,184 bytes(2.84GiB),
chunk64 700,328,960 bytes(0.65GiB), a 2.19GiB reduction in this fixture. Loss values
12.90292263/12.90292454. Single cold timings0.410/0.367s are retained without a
speed claim. Actual synthetic1024-token whole-model peak was11,976,679,424 bytes
(~11.15GiB), backward7.25s before optimizer step; it has no paired1024-token
whole-model baseline. The65-token pair had a slightly higher candidate peak and
cold runtime. Do not infer universal memory reduction, performance or convergence.

The initial `qwen35-cli-installed` run failed on the old target string check;
its logs/exit1 remain. No GPU fault or dmesg change. Final CLI logs have no MIOpen,
causal_conv1d fallback or allocator OOM warnings, and retain the legitimate
`chunk_gated_delta_rule` reference warning. Nothing was hidden with warning filters.

## Installation, recovery and remaining work

Deployed module SHA256:
`eddfea40c7fb2f0fd6dcfac72d67077918d1c61730ceba60d6e3f872f5e1f6d8`.
Final caller SHA256:
`03b0f285216ddd68a5948cbe21a759bb4fe78b9ab71d5ffda5725eb54d78f088`.
`SOURCE_QWEN35_TRAINING_OPS.json` records exact file/revision identities.
Source/patch/tests are local; original/final caller copies are retained locally
under ignored `kernel/artifacts/qwen35-training-reference/`. Host backup:
`WS/recovery/qwen35-training-before/{train.py,train.native-before-resume-fix.py}`.
Updates used guarded hashes and atomic copies under the existing runtime lock.
The project has no Git repository; none was initialized. No native C++ rebuild:
existing Triton cache and patched compiler/runtime were reused; ccache untouched.

Torch4ee65717d69e9f51021272e3f37d0f342f68fc08, libraries a6f7d3d, kernel0957f1/
image84995f1 and boot a287fd16-c97b-4d88-8055-b0553a2f7631 retained. No runtime DSO,
prefix selection, attention default, SDMA/experimental VM heap policy or boot
policy change. Math SDPA stays selected: this Qwen3.5 full-attention head width256
exceeds the separate optional tiled backend's128 limit. r2 fallback/BIOS/.105 and
user-owned TheRock src/ remain unchanged. No agent build/test/container/deadline
remains active at handoff.

Native flash-linear-attention/gated-delta training is **not implemented**. The
reference still executes GPU ATen math but has extra operations/allocations.
The global MIOpen EvaluateInvokers elapsed<=0 and GemmFwdRest workspace defect is
**not fixed**; this Qwen convolution bypass removes that consumer's solver search,
not other MIOpen failures. Missing CK/DB files were not faked or silenced.
The Python CPU segfault remains unresolved and separate from fixed VM-heap/CSR
corruption. Long-run stability, full-tuning, higher-order/compiled/graph paths and
longer sequences remain unqualified.

Exact next: profile reference gated-delta forward/backward on synthetic Qwen3.5
geometry and select one bounded native gfx1013 substage with CPU/output-gradient
references. Keep MIOpen root-cause isolation separate. N2.7 Schur iteration remains
queued behind this user-requested training follow-up.
