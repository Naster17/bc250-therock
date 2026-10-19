# N3.6: deployed trainer CLI at 1024 tokens/rank16/accumulation16

Completed 2026-10-19. Qualification code `bee5c6ce9`; no production application,
Torch, ROCm library, launcher or preset changed. The prior42-token/rank4 CLI
resume and N3.5 independent accumulation/AdamW evidence were inspected first.

The actual deployed `python -m replica.train` entry point ran inside the qualified
launcher environment on .104, with the current N3.2 GEMM preset/no preload.
This exercises the inner trainer CLI; the outer user profile wrapper was not
invoked and no user dataset, checkpoint or configuration was opened for training.
The preservation controller hashes the recorded protected paths without editing them.

## Results

Owned32 records each tokenize through the production formatter to exactly1024
input tokens. Assistant labels have32 distinct supervised counts72–289, with
context masked to-100; token IDs/masks/geometry are hashed in fixture.json.
Actual Qwen3.5-2B text base, BF16/autocast, FP32 adapters, rank16/alpha32,
batch1/accumulation16, math SDPA, non-reentrant checkpointing, AdamW and a
consistent two-update cosine schedule are used in both processes.

| Process/update | Input tokens | Supervised targets | Training loss | Evaluation loss | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh1 | 16384 | 2825 | 0.11412767701708111 | 0.16679230332374573 | 5284608000 |
| Fresh2 | 16384 | 2951 | 0.07038949553197377 | 0.1256762146949768 | 5419161600 |
| Resume checkpoint1 →2 | 16384 | 2951 | 0.07038949553197377 | 0.1256762146949768 | 5419161600 |

All372 decoded adapter tensors match exactly at step2, as do the complete saved
optimizer, scheduler, Python/CPU/GPU RNG, shuffled order, epoch/position/step,
run.json and training loss/token counts/LR. Evaluation2 equality is additionally
checked from the exported JSON logs. Seven composite gate checks pass.48 training
microbatches and three1024-token evaluation forwards executed. All losses finite;
no MIOpen search warning, causal-convolution fallback, OOM or traceback.
The native operation installer is visible in both logs.

Two guards acknowledge600s; whole-batch dmesg is byte-identical.309 protected
identities/four FFT AOT databases/links/runtime/user trainer/config unchanged.
Kernel `7.1.5-bc250rocm-r2`, same boot
`d3857b96-1b51-4e39-8f3e-f076f0c0422a`. No agent reboot, native build/install,
persistent cache reset, deletion, .105/BIOS/boot-fallback change. Runtime
`62ab2437c`, rocBLAS `8389197c1`, native Torch `dc8fea0`, Torch source `2b449c4`,
other native libraries `319a235d6` retained. No active agent test/container/guard.

This is bounded optimizer-boundary replay of synthetic masked data, not real-data
quality/convergence, sustained throughput, mid-accumulation recovery, compiled
training, CPU whole-model correctness or a new performance gain. Earlier N3.5
provides the independent FP64 accumulation and CPU optimizer references; the new
CLI comparison is uninterrupted GPU execution versus a fresh GPU process.
Historical Python CPU segfault remains unresolved and separate from the repaired
VM-heap/CSR corruption, N3.2 loader and prior N3.5 reference-fixture defects.

## Fingerprint audit and next task

Production run.json pins training arguments, data contents, model config/file
metadata, tokenizer and package versions; math retains the legacy fingerprint
schema. The current fingerprint does **not** pin numerical-runtime identity,
GEMM preset or application-hook content. This qualifier separately pins their
hashes and preserves the runtime throughout. Exact replay across changed preset/
libraries/hooks cannot be inferred from an accepted legacy fingerprint.
No old checkpoint or production resume policy was changed in this batch.

**Next N3.7:** close that provenance gap for newly saved checkpoints. Inspect
launcher runtime/preset/hook identities, design compatible supplemental checkpoint
metadata, qualify same-runtime replay and changed-runtime handling using owned
fixtures, and preserve explicit legacy checkpoint behavior. No arithmetic change
or silent rewriting of user checkpoints. Vocabulary-projection tuning remains
open after this reliability task; rejected square override remains undeployed.

## Reproduction and preservation

WS = `/home/nik/bc250work/rocm-10.2-gfx1013`.

```bash
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n36_on_host.py --attempt r1
```

Requires a new attempt/scratch directory, idle runtime and watchdog. CPU threads4;
explicit RAM-backed `/tmp/bc250-training-n36-r1` scratch requires>2GiB free tmpfs
and>6GiB available RAM. Like N3.5 this task alone opts into512MiB text-log disk
floor; ordinary controller default remains2GiB. Actual entry/final disk headroom
was~5.1GiB, no disk reclamation performed. Checkpoints/tokenizers/data/logs are
exported and SHA256-verified at local ignored
`kernel/artifacts/training-n36/scratch-r1`; remote /tmp copies are volatile.
All31 files (666855212 bytes) verified; see artifact-verification.json for every
copied file.

Raw WS/logs/training-n36-* and matching local
`kernel/artifacts/training-n36/remote`. The compact evidence below retains gate
results, guard/controller/preservation records and compressed kernel logs.
Executed r1 source archives and source-identity.json pin the raw test sources;
committed Python formatting was applied after staging. ASTs were compared equal;
Black/Ruff-F/syntax pass. The raw executed helpers remain on .104 and the formatted
sources are also preserved in the versioned memory snapshot. No source was
replaced while a gate was running.

Next preservation baseline `WS/logs/training-n36-qualification-r1/final.json`;
runtime recovery remains `WS/recovery/gemm-n32-before`. Previous N3.5 tmpfs and
durable local artifacts remain untouched.
