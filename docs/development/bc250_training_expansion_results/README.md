# Attention/training priorities1–4 — 2026-10-15

User ordered these tasks before returning to N2/K6. Native .104, SDMA0, existing
kernel/boot/SMU/fan policy and BIOS unchanged; .105 untouched. Math remains default.
Torch optimization43cb299 plus autocast/capability correction4ee65717d69e9f51021272e3f37d0f342f68fc08.
**Priorities1,2,4 passed within the scopes below. Priority3 remains unresolved.**

## 1. Longer attention and real model

`attention-long-p1`:11 output+Q/K/V-gradient cases, independent CPU float64 and GPU
math references; FP32/BF16 lengths257/513/1024 at Hq16/Hkv8/head128,2048/4096 at Hq2/
Hkv1/head128; FP16 at1024. Includes causal, explicit boolean/empty-row masks.
`attention-opt-long-p2` repeats these against the optimized candidate.

`attention-qwen-p1` and `attention-qwen-optimized-p4` use the existing local
Qwen3-1.7B-Base (1,720,574,976 base parameters; configSHA1bb33a92...), BF16,
rank4 q_proj/v_proj LoRA (802,816 trainable parameters), checkpointing, synthetic
512/1024-token inputs. Same model/adapter state, no optimizer mutation between
math/tiled reference passes. Final maximum per-adapter gradient relative L2
0.04075, sampled-logit relative L2<=0.00309, loss difference<=0.00211; recorded
bounds0.06/0.025/0.025. This is a bounded reference check, not convergence or a
claim that all gradients are identical. User data/adapters/checkpoints untouched.

Whole-model peak at1024 remained5,818,112,000 allocated bytes for both paths.
Tiled attention's isolated auxiliary-memory reduction does not imply the same
whole-model saving: other allocations, including full-vocabulary logits/loss,
dominate this fixture. Timings were unpaired short steps, not model-speed claims.

## 2. Optimize demonstrated causal work

Forward/dQ stop at min(key length, query row+1). dK/dV start at the first query
tile that can attend their key. Boundary tiles remain masked, including rectangular
causal and zero-length cases. Tile32 selected only for causal heads>=64; other
paths retain tile16. No matrix instruction or architecture spoofing.

`attention-opt-api-p2`:18 previous CPU/GPU reference cases plus padded Llama and
checkpoint/memory contracts pass. `attention-opt-bench-p2-r2`:same-input A1/B1/B2/A2,
nine GPU-event samples/pass, FP32/BF16, causal and control fixtures. BF16 head128
causal ratios original/candidate1.67–2.05 for these fixtures, with A2/A1 about
0.968–1.009. FP32 and controls show more drift, including an unchanged-code control
ratio0.848; do not advertise universal speedup or non-regression from those noisy
measurements. Initial tile32 noncausal FP32 candidate regressed; selection narrowed.
No sustained, whole-model or clock-attribution claim. Math default preserved.

## 3. Historical Python fault: investigation, not a repair

Recovered original event: python[8443], kernel uptime167348.960783, execution at
unmapped low IP0x3b3e6/error14. Many later kernel files contain this **same event**,
not independent reproductions. Original operator report stopped after49 checks
near sparse conversion/mm; no retained core/backtrace found at documented paths.
Do not infer a cause from this proximity or merge it with the proven old CSR
metadata corruption / experimental-heap default bypass.

`python-fault-sparse-p3`:eight fresh Python processes, debug Python/libc allocators,
240 CPU-referenced solver/FFT/CSR/COO/GEMM checks each. All1,920 pass; no reproduced
crash/core, kernel logs unchanged. Diagnostic-only core limit256MiB; per-operation
stage JSON and Python faulthandler enabled. Passing this does **not** fix the old fault.
The launcher now defaults PYTHONFAULTHANDLER=1, with explicit user override retained.
Direct launcher default-on and empty-value opt-out checked. Python treats the string
0 as nonempty/enabled; use `--env PYTHONFAULTHANDLER=` to opt out, not value0.
Normal launcher core limits unchanged. No OS-wide core-pattern/sysctl changes.

Separate failure retained: `python-fault-diagnostics-p3` full mixed55-operator
attempt stopped at its first convolution on MIOpen EvaluateInvokers elapsed<=0,
with GemmFwdRest workspace480/provided0 diagnostics. It exited1, **not SIGSEGV**,
and never reached the sparse area. That suite was not disabled or declared passed;
the separate sparse-focused diagnostic isolates the historical failure area.
This MIOpen issue remains open. Next: isolate that small-convolution timing/
workspace path; preserve a real crash/core/stack before proposing a CPU-fault fix.

## 4. Training/AMP/compile/resume

- Actual LoRA CLI defect reproduced in `training-app-resume-before-p4`: fresh run
  succeeds, resume raises RNG state must be a torch.ByteTensor after moving saved
  state to GPU. Resumed adapters also loaded frozen by default; scheduler state
  was absent. Patch `build_tools/bc250/patches/finetune-n7-resume-state.patch` fixes
  CPU RNG restoration, is_trainable=True, saves/restores scheduler/Python RNG and
  max-steps shuffle/remaining-index state. Existing script edits preserved.
- `training-app-resume-fixed-p4` actual CLI tiny disposable Qwen model/English data:
  fresh+resume both exit0,448 trainable adapter parameters, step2 checkpoint written.
  `training-legacy-resume-p4` resumes the original old-format fixture and verifies
  adapter weights change. Legacy missing scheduler/data-order state is reported;
  it cannot provide exact historical replay. Matching schedule arguments matter;
  restoring a finished cosine schedule does not create a new learning-rate plan.
- `training-amp-resume-p4-final`:two-layer random Qwen, **all parameters train**,
  batch2/length128 FP32/FP16 math, batch2/length256 BF16 math/tiled. CPU float64
  aggregate-gradient relative L2:7.08e-7/0.00138/0.00749/0.00728. Two finite updates;
  FP16 GradScaler skips no update. Fresh-process next update matches uninterrupted
  loss/model, optimizer buffers, scheduler/scaler and exact CPU/GPU/Python RNG probes.
  The earlier AMP tiled dtype mismatch is retained; wrapper now follows CUDA
  autocast lower-precision policy and gradients propagate through Q/K/V casts.
- `training-compile-p4-r2`:fullgraph/dynamic math-attention Transformer block,
  batches2/3, lengths128/96, FP32/FP16/BF16 output/input/parameter-gradient equivalence.
  Generated-kernel counters checked, suppress_errors=False. Maximum relative L2
  <=2.95e-7/0.000957/0.001886. This does not add compiled tiled attention or compile
  an entire Qwen3-1.7B model. Fixed a false TF32 Tensor Core warning by using the
  existing native is_tf32_supported API instead of CUDA-major-number comparison;
  the legitimate unsupported max-autotune warning is retained.

## State and replay

All final successful labels exit0,600s recovery guards ACKed, before/after kernel
logs identical. Failed mixed/AMP/app/harness runs retained. Helpers run through
`WS/helpers/update_dev_prefix.py -- python3 WS/helpers/validate_tiled_sdpa_on_host.py`
with a unique --label, --installed --script NAME and --test-arg=ARG for each argument.
Diagnostic core capture is explicit --capture-core, not a normal launcher default.
WS=/home/nik/bc250work/rocm-10.2-gfx1013. Raw kernel logs and disposable model/
checkpoint fixtures stay in WS/logs/LABEL; this directory retains JSON/log evidence,
not weights/tokenizers. Runtime and source identity: SOURCE_TRAINING_EXPANSION.json.

Source changes are Python-only after the prior Torch probe DSO build; no new
native build needed. Existing Triton/Inductor caches reused; safe compiler defaults
retained. Rollback copies: WS/recovery/attention-priorities-before,
training-expansion-before, tf32-diagnostic-before and training-expansion-metadata-before.
Every active-prefix mutation used the exclusive lock. No agent workload remains.
N2/K6 remain deferred by the user's instruction; historical CPU fault and newly
observed MIOpen failure must remain visible rather than marking all four fixed.
