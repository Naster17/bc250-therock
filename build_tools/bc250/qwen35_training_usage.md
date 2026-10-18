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
