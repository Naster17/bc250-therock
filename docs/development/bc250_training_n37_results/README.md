# N3.7: checkpoint numerical-runtime provenance

Completed 2026-10-19. Code `e55d8b23a` (application implementation `0f2dfa078`).
The deployed trainer now supplements the existing run fingerprint with
`runtime-provenance.json`. Its canonical SHA256 binds both run.json and training.pt.
The original fingerprint/config schema and training arithmetic are unchanged.
New checkpoints refuse absent/corrupt metadata or a changed recorded runtime.
Legacy checkpoints retain their original checks and explicitly warn that their
numerical-runtime continuity is unverified; they are never rewritten. New saves
from a legacy resume record the current runtime.

## Identity and compatibility boundary

The record includes130 selected file hashes: trainer/formatter/provenance/hooks,
Torch native/Python training modules, selected ROCm DSOs, Qwen3.5 implementation
and110 gfx1013 rocBLAS assets.15 actual mapped core native-library identities are
also hashed. Active preset bytes, selected numerical environment variables,
Python/kernel/GPU architecture and Torch/HIP versions are recorded. Paths to the
application/preset are not used as content identity, so the candidate overlay and
installed application can compare exactly.

Collection runs once per process before model weights load, adding startup hash
I/O. No per-update GPU arithmetic is added. The sidecar digest is checked before
preflight returns; live runtime comparison follows GPU availability/property
checks, before weights/output creation. Data-only preflight does not initialize
the GPU and does not claim live runtime equivalence. The training-state binding
is separately verified when trusted training.pt is loaded. This is an audit aid
for selected eager Qwen3.5 paths, not a security boundary, complete image/package
integrity or proof of determinism/convergence. Whole-source hashes conservatively
reject even nonnumerical source edits. There is no silent migration or bypass;
restore the checkpoint's recorded runtime/preset/hooks when a new checkpoint
reports a mismatch.

Integration is a minimal reviewed patch to the recorded production trainer;
full candidate source, module and reconstruction patch are tracked under
build_tools/bc250. The imported trainer's existing formatting is preserved.
Updated usage is installed at `/home/nik/finetune-n7/BC250_TRAINING_OPS.md`.
Existing outer wrappers already dispatch to this trainer and need no command change.
This does not alter generic torch.save or unrelated applications.

## Qualification

Actual Qwen3.5-2B text model, owned32 records of exactly1024 input tokens,
32 distinct assistant target masks72–289, batch1/rank16/alpha32/BF16 base and
FP32 adapters, math SDPA, non-reentrant checkpointing and accumulation16.
Same two-update cosine schedule and current N3.2 GEMM preset/no preload.

- Owned source overlay: two fresh updates, separate new-checkpoint resume1→2,
  then real legacy resume from the unchanged owned N3.6 checkpoint1. All372
  decoded adapters/full optimizer/scheduler/CPU+GPU+Python RNG/order/progress/
  run.json/loss/token counts/LR match exactly. Legacy warning appears only for
  that legacy process; its new checkpoint has bound current provenance.
- Eight CLI refusals: missing sidecar, malformed JSON, corrupted digest,
  unsupported schema, orphan metadata, changed preset, changed hook source and
  simulated changed recorded native-file identity. All exit2 before model
  weights load and without creating output. No native DSO was modified.
- Installed trainer resumes the same owned checkpoint: all372 adapters/full state/
  run.json/provenance exact against candidate control. Its numerical state also
  matches the pre-N3.7 control exactly after excluding only the new metadata key.
- Evaluation2 logs agree exactly across fresh/new resume/legacy/installed runs:
  loss0.1256762146949768. Update2 loss0.07038949553197377,
  targets2951/input16384/peak5419161600B. Fresh1 loss0.11412767701708111.
  These finite synthetic updates are not a convergence or speed comparison.

80 training microbatches and five evaluation forwards executed.26 qualified
composite checks (14 retained training +8 corrected refusal +4 installed) pass;
eight initial refusal checks remain separately recorded in a failed kernel-log
gate. Four CPU contract tests pass locally and on .104. Black on eight new
helpers, Ruff-F on helpers/imported trainer, syntax and patch dry-run pass.

### Preserved guard failure and correction

Original r1 training/new-resume/legacy gates pass with byte-identical dmesg.
All eight r1 refusal assertions also pass, but its controller correctly fails
because the kernel log gains exactly two network initialization messages:
XFRM netlink socket and bridge netfilter notice. The failed gate/whole r1 batch
remain failed in the evidence; their cause is not attributed to a GPU defect.
No GPU error appears in that delta.

The recheck validates the original protected/source identities, all three passed
training gates and that exact two-line delta. It reruns only the refusal gate
in fresh owned scratch, requiring byte-identical dmesg again. r2 passes;
no GPU model training is repeated to erase the r1 failure and no guard criterion
is relaxed. r3 guarded installation/replay also passes with unchanged dmesg.
Six600s guards acknowledge. Whole-batch dmesg is therefore **not** byte-identical;
only the documented two-message delta is present. All failed records retained.

## Deployment, preservation and resources

Installed under the exclusive update_dev_prefix.py lock after candidate gates.
Intended app changes only: replica/train.py, new
replica/bc250_checkpoint_provenance.py and the usage guide. Old trainer/guide
backed up at `WS/recovery/training-n37-before-r3`; deployment records preserve
before/after hashes. The installer restores the old app if installation or its
post-deployment gate fails. No rollback was needed. Of309 original protected
paths only train.py changes intentionally;308 remain identical. Module/guide
are added to the preservation set, final311 paths match the new expected state.
Four FFT AOT databases/defaults/links/user profile/model/data/checkpoints remain.

Runtime62ab2437c, rocBLAS8389197c1, native Torchdc8fea0/source2b449c4 and other
native libraries319a235d6 retained. Same r2 kernel/bootd3857b96. No native build/
install, cache reset, deletion, reboot, .105/BIOS/fallback edit. Unknown Python
CPU segfault remains unresolved and separate from fixed VM-heap/CSR corruption,
repaired N3.2 loader and this network-log guard failure.

Disk headroom observed~5.1GiB at entry and~1GiB near completion; no reclamation
or reserve change. Like N3.5/6, only this batch opts into512MiB text-log disk floor;
ordinary controller default stays2GiB. Startup requires>6GiB available RAM and
>2GiB tmpfs scratch. Checkpoints/owned fixtures use explicit host tmpfs mounts.
The original N3.5/6 scratch/checkpoints/local copies are unchanged.

WS = `/home/nik/bc250work/rocm-10.2-gfx1013`. Commands, as root under the existing
exclusive lock, were:

```bash
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n37_on_host.py --attempt r1
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_provenance_recheck_n37_on_host.py --attempt r2
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/deploy_training_n37_on_host.py --attempt r3
```

The correction/deploy scripts intentionally refer to their recorded r1/r2
fixtures and are not generic rerun commands. Fresh qualifier requires new scratch.
Raw WS/logs/training-n37-* and local ignored kernel/artifacts/training-n37/remote.
RAM `/tmp/bc250-training-n37-r1` and `/tmp/bc250-training-n37-refusals-r2` are
volatile across reboot; durable local copies are kernel/artifacts/training-n37/
scratch-r1 and scratch-refusals-r2. Regular file SHA256/size and symlink targets
are verified without copying the symlinked user sources/checkpoints repeatedly.
All107 regular files (1111909408B) and159 symlink targets verified.
See artifact-verification.json. Next preservation baseline
WS/logs/training-n37-qualification-r3/final.json (311 paths); app recovery above,
native runtime recovery remains WS/recovery/gemm-n32-before. No active own
job/container/guard.

**Next N3.8:** return to vocabulary-projection tuning: inspect N3.3 actual
forward/backward layouts and design chunked CPU-double references with bounded
RAM/VRAM before testing admitted scalar solutions. Keep current N3.2 preset in
both sides; unchanged full-model gradient/optimizer acceptance before deployment.
The rejected square override stays inactive. No universal performance claim.
