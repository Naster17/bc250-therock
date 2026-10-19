# BC-250 Qwen3.5 LoRA operations (2026-10-18)

`replica/train.py` installs `replica/bc250_ops.py` after moving the text-only
Qwen3.5 model to native gfx1013. The existing `scripts/train_bc250.py` command
therefore enables these operations by default. Transformers is pinned to the
project's reviewed 5.19.0 overlay; other versions are refused by the installer.
No new package installation or profile/checkpoint format migration is required.
The source SHA256 is printed once in each trainer process.

- Frozen depthwise causal convolution, K<=4, FP32/FP16/BF16: native scalar GPU
  forward and input backward with optional SiLU. This avoids MIOpen search for
  this operation. Trainable convolution weights/bias and unsupported inputs keep
  the original implementation. Cached single-token convolution update is unchanged.
- Loss: token chunks of64, FP32 cross entropy and recomputed probabilities in
  backward. Label shifting, -100 masking, mean and explicit item counts are
  preserved. Full BF16 logits still exist; this is not fused linear cross entropy.
- Gated-delta: precision-safe full GPU operation from corrected A06. Low-precision/
  autocast preparation/scan matches reference ordering; native FP32 preparation
  remains. `BC250_GATED_DELTA=0` explicitly restores the Transformers reference.
  Complete fused FLA is still absent; no fake package or warning filter is installed.
- Full attention keeps math by default. Inner and outer trainer accept
  `--attention-backend tiled` for a new output/run. Head256 BF16 training uses
  math forward/native tiled backward: lower isolated peak memory but slower in
  measured fixtures, with quadratic forward temporary allocations. Changed backend
  has a distinct checkpoint fingerprint; default math keeps its legacy schema.

Scope: eager first-order LoRA with frozen convolution weights, math or explicit tiled/hybrid SDPA and
non-reentrant gradient checkpointing. Full tuning of convolution weights,
compiled/graph execution, higher-order differentiation, decode optimization,
longer sequences and long training convergence have not been qualified here.

To compare with the original operations, copy your reviewed training.env to a
new file, add `BC250_QWEN35_TRAINING_OPS=0`, then pass that file using the existing
`scripts/train_bc250.py --env-file /absolute/path/to/copied.env` option. A direct
`bc250-train` call can pass `--env BC250_QWEN35_TRAINING_OPS=0` before its script.
The regular launcher does not inherit arbitrary caller environment variables.

Existing checkpoints and run fingerprints are preserved; this backend change
can change floating-point rounding and does not promise bitwise continuation
of a run that used the original operations. Resume with the original model,
data, tokenizer, profile and schedule. No old checkpoint is rewritten.

Tests use synthetic inputs only: CPU-float64 operator references, independent
ATen GPU tap-sum convolution plus original Transformers loss for LoRA gradients,
and actual downloaded text-only Qwen3.5-2B forward/backward and adapter updates.
Isolated1024-token/vocab248320 BF16 loss extra peak allocation was2.84GiB for the
reference and0.65GiB for chunk64. This is not a whole-model memory or speed claim;
full logits and reference gated-delta attention still consume substantial memory.

Corrected A06/A07: all372 nonzero BF16 model gradients exact for A06 at65/1024;
A07 stays inside unchanged4% per-adapter reference limits. Exact fresh/resume
continuation passes within each backend. This does not promise convergence,
universal speedup or bitwise continuity across a code/backend change. See
`therock/docs/development/bc250_a06_a07_results/README.md` in the workspace.

## Optional GPU profiling (A09, 2026-10-18)

The selected .104 Torch build now supports Kineto/rocprofiler-sdk GPU traces.
Use explicit `torch.profiler.profile` contexts with CPU and CUDA activities;
ROCm uses the CUDA activity API name. Export a Chrome trace to your persistent
project directory, then inspect kernel/copy events, CPU correlations and optional
allocation/free records. `self_device_time_total` sorts GPU timings in tables.
The trainer does not automatically collect profiles, and profiling adds overhead.
Hardware counters, simultaneous external profilers, compile/graphs and long
traces remain unqualified. The workspace guide
`therock/docs/development/bc250_profiler_a09_results/README.md` contains examples,
exact references, measurements and recovery instructions.

## Current-preset CLI resume qualification (N3.6, 2026-10-19)

The deployed inner trainer passes an owned synthetic1024-token/rank16/
accumulation16 test with variable assistant target masks: checkpoint1 resumed in
another process reproduces update2 exactly for all372 adapter tensors, complete
optimizer/scheduler/RNG/data progress and training/evaluation losses. This is a
bounded optimizer-boundary test, not long training quality or a new speed claim.

Keep the numerical runtime, GEMM preset and application hooks consistent when
resuming. Existing fingerprints cover data/model/settings/package versions but
do not pin those additional identities; the test records them separately.
Changed-runtime exact replay is unqualified. N3.7 adds the supplemental check
below for newly saved checkpoints; existing user checkpoints remain untouched.
Evidence: `therock/docs/development/bc250_training_n36_results/README.md`.

## Checkpoint runtime provenance (N3.7)

New checkpoints contain `runtime-provenance.json`. Its digest is stored alongside
(the unchanged) training fingerprint in `run.json` and in `training.pt`. It records
selected native Torch/ROCm libraries, gfx1013 rocBLAS assets, the active GEMM preset,
application hooks, selected numerical environment settings, Python/kernel/GPU
architecture and version information. Collection happens once at process startup;
normal updates and their arithmetic are unchanged.

On resume, new checkpoints reject a missing/corrupt sidecar or a changed recorded
runtime identity. Runtime mismatch is checked before loading model weights or
creating the output directory. Restore the matching runtime, preset and hooks;
do not remove provenance to bypass the check. A matching identity is an audit aid,
not proof of universal determinism, complete image integrity or convergence.

Legacy checkpoints without provenance still use their existing fingerprint and
state checks and print an explicit warning that numerical-runtime continuity
cannot be verified. They are never rewritten. New checkpoints saved by the resumed
run record the current runtime. Data-only `--preflight` validates the sidecar and
binding but defers comparison with the live numerical runtime until training;
it does not initialize the GPU or load weights.

Whole source files are part of that identity: even a changed status message can
refuse a provenance-bound checkpoint. N3.9 restores the N3.7 trainer/provenance
and explicit tiled CLI while preserving the current ops helper's status edit.
Checkpoints saved with the older helper require its recorded source/runtime;
they are not silently migrated. Matching new checkpoints and legacy checkpoints
are qualified separately. Math remains the default; the N3.8 vocabulary GEMM
candidate is experimental and is not installed.
