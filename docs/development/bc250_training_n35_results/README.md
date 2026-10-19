# N3.5 accumulation16 and checkpoint replay — 2026-10-19

TheRock code `986093a3c` adds actual-geometry training qualification, RAM-backed
artifact mounts and an isolated CPU optimizer reference. No Torch/library build,
runtime/preset/application change. Installed runtime62ab2437c/rocBLAS8389197c1,
native Torchdc8fea0/source2b449c4/other native libraries319a235d6 remain.

## Scope and results

Actual local Qwen3.5-2B base, owned nonzero rank16/alpha32 adapters, synthetic
1024-token IDs, batch1, accumulation16, BF16 base/autocast, FP32 adapters,
math SDPA, checkpoint recomputation, dropout0. Labels are all input tokens;
no real dataset, variable target masks or production trainer CLI is exercised.
Historical42-token/rank4 actual-CLI resume evidence remains distinct.

Sixteen independent scaled microbatch gradients are summed on CPU in FP64.
Restore initial CPU/GPU/Python RNG, then perform ordinary GPU FP32 accumulation
with loss/16 and identical losses, IDs and three random probes per microbatch.
All372 accumulated gradients pass fixed atol1e-8/rtol1e-5 and relativeL2\<2e-6;
worst relativeL2 **1.1708998609406108e-8**. This independently checks accumulation,
not a CPU reference for the whole model's forward/backward arithmetic.

Global clipping has an independent FP64 norm/scale reference. First/second
uninterrupted norms CPU6.0788796339/5.2676186834 versus GPU6.0788798332/5.2676186562.
Each update checks all372 parameters and AdamW step/first/second moments against
its own CPU FP32 replay: parameters atol2e-7/rtol1e-5; state atol1e-12/rtol1e-5.
Maximum parameter abs errors1.863e-9/3.725e-9. LambdaLR state matches CPU replay;
next learning rates8.5355339059e-5/5e-5. No new tolerance adjustment.

Save an owned optimizer-boundary checkpoint after16 microbatches/one update.
Uninterrupted step2 and a fresh process restored from that checkpoint match
**exactly** in372 parameters,372 clipped gradients, full AdamW/scheduler state,
16 losses/input digests/CPU+GPU+Python random probes, final RNG and progress
(two updates/32 microbatches). Resumed own-gradient CPU AdamW maxabs1.863e-9.
Archive files have different serialization bytes; exactness refers to decoded
state/tensors. Mid-accumulation resume, real labels/data order, dataset throughput,
convergence and general full-model correctness remain unqualified.

## Preserved test-fixture failures and reuse

- r1 stops before model work: adding pinned consumers overwrote the explicit
  scratch mount. Test launcher now appends that mount; normal default mounts
  remain identical. Failure/source/log/guard retained.
- r2 completes independent16 + accumulation16 + uninterrupted16. Fresh resume
  then fails the CPU optimizer check because loading the same checkpoint into
  GPU and CPU optimizers aliases the CPU `step` tensor. Pinned Optimizer loader
  deliberately retains noncapturable step tensors; this is caller-owned reference
  state, not a demonstrated Torch checkpoint bug. CPU oracle now loads a deep
  tensor clone and asserts counters do not alias. No production patch needed.
- r3 reuses the six passed r2 checks and unmodified checkpoint/control/reference
  artifacts. Original source/protected hashes, progress/counters and input file
  hashes are verified. Only fresh16 is repeated with the isolated CPU oracle;
  decoded final state matches exactly. The original qualification-file SHA is
  explicitly admitted from its recorded source identity while the actual corrected
  test SHA is separately recorded. Model/helper/app/preset identities are unchanged;
  no checkpoint contents, numerical contract or GPU arithmetic changed.

Composite eight checks pass.80 actual model microbatches executed across r2/r3:
64 constitute the reference/accumulation/control/corrected-resume proof,16 belong
to the retained failed CPU-oracle branch. Three guards ACK600s; initial two
fixture failures preserved. Dmesg byte-identical across all attempts;309 protected
identities/four FFT AOT DBs/links/runtime/preset/user trainer/config unchanged.
CPU threads4, no native build/persistent cache reset, .105/BIOS/fallback change.
Historical Python CPU fault remains unresolved/separate from fixed VM-heap/CSR,
repaired N3.2 loader and these test-fixture defects.

## Changed entry state and artifact capacity

At entry .104 had already rebooted to r2 `7.1.5-bc250rocm-r2`, boot
`d3857b96-1b51-4e39-8f3e-f076f0c0422a`; earlier d0532075 boot is historical.
All309 prior hashes still matched. Disk had only881MiB free. No file/cache/image
was deleted or filesystem reserve changed. New N3.5 preflight requires /tmp tmpfs,

> 2GiB scratch and>6GiB available RAM. Explicit opt-in512MiB disk floor applies to
> these text-log-only gates; other controllers retain2GiB default. Checkpoints/
> reference tensors live in `/tmp/bc250-training-n35-r2`, ~836MiB; they are volatile
> across reboot. All four are durably copied/hash-verified at local ignored
> `kernel/artifacts/training-n35/scratch-r2`; preserve this copy. Remote sources/
> compact logs remain in WS/logs/training-n35-\*; matching local ignored
> `kernel/artifacts/training-n35/remote`. Source attempts are archived here.

WS = `/home/nik/bc250work/rocm-10.2-gfx1013`. Replayable commands (replace WS;
fresh attempt/scratch directories required):

```bash
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n35_on_host.py --attempt r2
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_replay_n35_on_host.py --attempt r3
```

Current full qualifier includes both fixes; this batch validates retained controls
plus fresh replay, rather than rerunning its entire sequence.
The retained-replay helper intentionally targets the existing r2 fixture.
Next preservation baseline WS/logs/training-n35-qualification-r3/final.json;
helper backups WS/recovery/training-n35-before; runtime recovery remains
WS/recovery/gemm-n32-before. No running agent job/container/guard; disk~880MiB.

**Next N3.6:** qualify the actual deployed trainer CLI with owned1024-token/
rank16/accumulation16 synthetic data and fresh-process resume on the current
preset. Audit prior small CLI tests and production checkpoint fingerprints first;
keep user datasets/checkpoints/config untouched and use explicit RAM-backed
artifacts/local durable copies while disk headroom is low.
