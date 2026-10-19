# N3.9: trainer feature reconciliation

2026-10-20 local (2026-10-19 UTC). Qualification implementation `9899e0659`,
historical-checkpoint refusal `5a24fe4e7`. Restore the byte-exact qualified N3.7
trainer (`n37_app_train.py`, SHA256
`b9bf2587d8f3b14b3f7ca7f62bcfface0c1a72606633ac5920600c28f1f0f4c1`)
after reviewing every difference against the preserved N3.8 entry source.
The reconstruction patch adds only the missing N3.7 provenance integration and
A07 attention-backend CLI/selection. Current ops helper/status edit remains
unchanged (`d058b14a...`); it is arithmetically identical to the old qualified
hook. No user training data or checkpoints participate in qualification.

Math remains the default, and its legacy fingerprint schema is preserved.
Explicit `--attention-backend tiled` retains a distinct fingerprint; this uses
the existing head256 BF16 math-forward/tiled-backward hybrid, not FlashAttention.
The native N3.2 preset stays active. N3.8 vocabulary candidates remain inactive.
No new native implementation or speedup is asserted by this restoration.

## Tests and continuity contract

Reuse the N3.7 actual CLI contracts with owned **disk-backed** scratch at
`WS/experiments/training-n39-r1`. The shared validator gains an optional allowed
scratch parent; default `/tmp` behavior is unchanged. Its previous source is
preserved at `WS/recovery/training-n39-before/validate_training_n37.py`.
No native build, ccache reset or dependency installation is needed.

- Actual masked32-record fixture: every input exactly1024 tokens,32 distinct
  target counts72–289, batch1/rank16/alpha32/BF16 base/FP32 adapters/math SDPA/
  non-reentrant checkpointing/accumulation16. Fresh two updates; separate new
  checkpoint1→2 resume; owned N3.6 legacy checkpoint1→2 resume. Compare all372
  adapters/full AdamW/scheduler/CPU+GPU+Python RNG/order/progress/run.json/loss/
  token counts/LR, with bound provenance on new saves and an explicit legacy warning.
- Eight metadata/runtime refusals preserve the original N3.7 failure boundary:
  absent/malformed/corrupt/future-schema/orphan record, changed preset/hook and
  simulated changed native identity. Exit2 before weights/output; no real DSO edit.
- Short actual tiled CLI fresh2/resume1: all372 adapters/full state and bound
  provenance exact within that backend. Preflight verifies unchanged old-math
  fingerprint versus current entry, distinct tiled fingerprint, and refusal of
  a tiled checkpoint with math settings. This is a short sequence integration
  check; the original operator/model A07 evidence remains separately bounded.
- An actual owned N3.7 provenance checkpoint must refuse the preserved current
  ops source edit before weights/output. Whole-source identity conservatively
  includes printed status. This is intentional, not a migration: old checkpoints
  require their recorded hook/runtime. The owned historical checkpoint is hashed
  before/after, and remains unchanged.

The application initially has174 tests with three pre-existing errors: the
launcher fixture lacks `attention_backend` although the outer launcher already
reads it. Correct only that fixture default and add one test covering default,
profile and explicit math/tiled forwarding. Candidate suite passes175 tests.
The documented `.venv/bin/python` is absent on .104; use existing `python3` for
these standard-library tests. Four provenance CPU contracts also pass locally
and on .104. No Python/Torch package is installed to run the suite.

Installation is limited to the reconciled trainer, corrected test fixture and
usage guide. Back up each existing file, verify all protected entry/source
identities, run the full application suite, then replay an owned checkpoint
through the installed trainer under the existing exclusive runtime lock.
On installation/post-installation failure, restore those original files and
retain the failed candidate in recovery. Current provenance module/ops/native
runtime remain unchanged. Never rewrite user checkpoints or silently remove
provenance to make a resume succeed.


## Final results and preservation

Candidate r1 passes26 composite checks; r2 actual historical-checkpoint refusal
passes1; guarded r3 installation/replay passes4. **31 composite checks pass**.
Installed replay reproduces all372 adapters/full state/run.json/provenance and
loss/evaluation exactly; after excluding only the metadata marker, numerical
state also matches the pre-N3.7 control exactly. Step2 loss0.07038949553197377,
masked targets2951/input16384; evaluation loss0.1256762146949768/72 targets.
83 GPU training microbatches and eight evaluation forwards executed across
these bounded tests. No real-data training/convergence or throughput claim.

All175 application tests pass against both candidate and installed trainer.
Black on new qualification helpers/shared validator, Ruff-F, syntax, and patch
dry-run plus byte-exact reconstruction pass. Imported app test formatting is
preserved. Four provenance CPU contracts pass locally and on .104.

Seven600s runtime guards acknowledge; all GPU gates and whole qualification
batches require byte-identical dmesg and pass. No GPU/host OOM or CPU segfault
occurs in this batch. The historical CPU fault remains unresolved, separate
from repaired VM-heap/CSR and N3.8's retained host OOM/SIGKILL failure.

The post-installation r4 documentation-only step corrects only the stale
attention paragraph of application AGENTS.md: math default, qualified explicit
tiled/hybrid scope and conservative checkpoint identity. No GPU qualification
is inferred from this docs step. Existing instruction text otherwise stays.
Final313 protected paths:309 of311 original paths unchanged, trainer/usage guide
intentionally updated, test fixture/application AGENTS newly protected. Ops/
provenance module/four FFT AOT databases/native libraries/runtime links remain.
Backups and failed-candidate rollback are implemented in
`WS/recovery/training-n39-before`; no rollback was required.

Runtime62ab2437c/rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4/other libraries
319a235d6 retained. Same7.1.5-bc250rocm-r2/bootd3857b96. No native build/install,
cache reset, deletion, reboot, BIOS/fallback or .105 change. Final observed disk
available5,490,642,944B. Own jobs/containers/guard deadlines finished; root SSH
closed. Evidence export and memory snapshot are separate preservation work.

Next **N3.10:** read prior compiled-block/attention evidence and inspect actual
Qwen3.5 nonzero-LoRA MLP dispatch, then qualify eager/compiled outputs/input and
adapter gradients using bounded independent references. Keep N3.2/math defaults;
full-trainer compilation and performance require separate evidence. Do not
activate the variable N3.8 vocabulary preset as part of this work.


All257 regular/resolved-source files (1,327,912,905B) are SHA256-verified durable
locally in `kernel/artifacts/training-n39/export`: raw host logs, all owned
checkpoints, synthetic data, application/test source copies and recovery files.
40 external Python source links were resolved to copied bytes;57 owned fixture
links were reconstructed inside the local export and verified. User data was
not followed through arbitrary links. Old N3.5–N3.8 artifacts remain preserved.
Compact raw results, compressed logs, hash lists and verification are beside this
report. Remote scratch is on disk, not volatile tmpfs; historical N3.6/N3.7
controls stay untouched. Latest preservation baseline is
`WS/logs/training-n39-guidance-r4/final.json` (313 paths); GPU installation
qualification is `training-n39-qualification-r3/final.json`. Commands:

```
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/prepare_training_n39_on_host.py
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n39_on_host.py --attempt r1
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_checkpoint_continuity_n39_on_host.py --attempt r2
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/deploy_training_n39_on_host.py --attempt r3
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/reconcile_training_guidance_n39_on_host.py
```

`WS` is `/home/nik/bc250work/rocm-10.2-gfx1013`. These one-time preparation/
installation commands intentionally refuse changed sources or existing scratch/
backup directories; inspect their recorded artifacts before any replay. Do not
reuse the same attempts or apply restoration twice to an already restored app.
