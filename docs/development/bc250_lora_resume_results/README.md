# L02: real two-host LoRA checkpoint restoration

Completed 2026-11-07. Checkpoint/replay implementation `284d3315b`; Docker-aware
guardian, negative controls and final verifier `4d5f60e43`.
[Verified metadata](verified-results.json),
[commands](../../../../BC250_LORA_TWO_HOST.md#restore-a-sealed-l02-checkpoint).

The separately authorized `bc250-projects/lora-smoke` application now has an
additional `train_two_host_l02.py`, `run-distributed-l02`, checkpoint support and
standard-library `checkpoint_files.py`. Both hosts have a fresh-run
`two_host_training_l02.json`. L01 entry points, completed outputs, native
checkpoints, model files, environments, native providers and boot fallbacks remain
intact. No `finetune-n7` access, native build/install or prefix mutation.

## Implementation and supported boundary

Native HF/PEFT/Trainer loaders still restore the adapter, AdamW, scheduler, scaler
and rank RNG. Added `bc250-l02.json` seals completed native checkpoints on both
ranks with a shared ID, common file hashes, rank-local RNG hashes, model/data/
settings fingerprints and full state hashes. TRL 1.17.1 resets its cumulative
`_total_train_tokens` in a new process; the integration now saves/restores that
counter. Native aggregate `train_loss` remains a current-process statistic;
per-step metrics are the exact replay comparisons.

The offline transport exports only rank 0's common payload, matches it to the
peer's own checkpoint marker/RNG, validates a fresh staging directory, then
publishes it by atomic rename. Original checkpoints and RNG files are unchanged.
It refuses incomplete/corrupt files, mixed generations, symlink payloads,
existing destinations and unsupported world sizes. It never unpickles payloads.
The bounded payload limit is 3 GiB; this is not a general replicated or
crash-durable checkpoint service.

Pinned application sources: Transformers 5.22.1, TRL 1.17.1, Accelerate 1.15.0
plus the L01 support module; PEFT 0.21.2 is the tested installed version. Supported
entry-point scope is two ranks, FP16, native AdamW, workers=0 and unpacked batches.
Settings/data/model contents must match; output paths may differ. Old unsealed
or one-host checkpoints, world-size changes, different training lengths,
partial-window recovery, shared-filesystem use and other models remain unqualified.

## GPU reference and replay

Actual Qwen3.5-2B, rank128/alpha256/all-linear, 134,553,600 trainable FP32 elements
in372 tensors, FP16 base/autocast, eager attention and non-reentrant checkpointing.
Same global16 as L01: micro2, accumulation4 per rank; 128-record input subset,
seed42, dropout0.05, max-length8192, four Trainer windows, cosine LR1e-4 with one
warm-up window. Input and shifted-label hashes are recorded without message text.

R2 reference saves checkpoint2, then continues uninterrupted through windows3/4.
Fresh workers load that checkpoint on each host. Every loaded parameter and all
372 AdamW states/counters, scheduler and scaler hashes match the saved state.
The test starts the resumed scaler at8; native loading restores512. It poisons
Python, NumPy, CPU Torch and GPU Torch RNGs after `on_train_begin`; native Trainer
restores all four correctly after its iterator/data skip, before the next forward.
The saved TRL token count10648 restores and continues to15745/19377.

Full window3/4 transcripts compare **exactly** on each rank: data hashes/counts,
all gradient bytes, trainable weights, all AdamW state, scheduler/scaler, four RNG
hashes and per-step loss/norm/LR/entropy/accuracy/token metrics. Final DDP replica
weights agree. Namespace PIDs/hostnames can be reused; different kernel process
start times and UUIDs establish fresh workers. This is a same-graph checkpoint
replay, not an independent CPU Qwen numerical reference or full-epoch proof.

For this probe only, native scaler growth interval1 produces scales
128→256→512→1024, then a natural overflow on window4 backs off to512. Native AdamW
is called for the first three windows (first LR0), and skips window4. Parameters,
optimizer and scheduler remain unchanged on the skip; data/RNG are consumed as
in the reference. The same skip, including invalid-gradient hashes, replays
exactly. Production retains the native scaler growth interval.

## Guard results and retained failures

Final comparison uses clean R2 reference guards on November6 and clean R3 resume
guards on November7, after external/user board reboots. No reboot was initiated
by these tests. R3 checks both Podman and the local default Docker daemon before
and after launch; kernel logs and registries remain unchanged, no foreign GPU
handles or containers were admitted. Existing runtime shared locks and the
4-CPU/cpuset0–3/10GiB RAM/12GiB total/512-PID/256MiB shm/600s recovery/460s child
envelope were retained. No full training job was started.

Eight guards are retained: **five accepted, three rejected**, all recovery guards
acknowledged. Two R1 references were rejected by an overstrict probe assertion
that initially disallowed the legitimate native overflow skip; the probe was
corrected to verify skip preservation/backoff. R2's first resume passed the exact
math on both nodes, but .105's system guard correctly failed when unrelated Docker
startup changed kernel logs with veth/bridge and journald messages. That failure
is retained, not relabeled. Docker detection was added and a clean pair repeated.
The user authorized stopping `steam-headless` and `brave_engelbart`; both were
already stopped when final testing resumed. No unrelated container was stopped
by the agent. Older unrelated guardians still need the Docker-aware check before
their next exclusive GPU experiment.

Native recovered allocator retry messages in the reference remain recorded;
this does not qualify every8192-token sample or improve capacity. Reference
causal-convolution/gated-delta warnings remain. Historical CPU Python/KFD issues
are distinct from fixed VM heap corruption, D05 agent exit and D21 optimizer load.

## Evidence and next action

Fifty-four local tests passed: the previous42 plus8 checkpoint-transport and4
Docker-preflight tests. Three real-checkpoint controls on .105 reject incomplete
payload and mixed generations while preserving original RNG/no failed destination.
Black, critical Ruff+F401, Python syntax, shell syntax and diff checks pass; no
full upstream CI. No throughput/convergence gain is claimed by L02.

Common payload: **11 files /1,635,023,695 bytes**. Original checkpoint2 is retained
on .104 and the original rank1 RNG checkpoint on .105; the prepared full .105
copy is `outputs/two-host-tests/l02-r2/ready/checkpoint-2`. Transport archive
SHA256 `fd4381b63ebe197eb455fa336e546224e28f5a30cd4e889313f310bc20378478`.

Workstation `kernel/artifacts/lora-resume/{104,105}-evidence.tar.gz`:

- .104: 220526 bytes, SHA256 `c4e6bbe17a4c2edf975493f512a4fc5197a26deaa32911a50f78f87859eca890`.
- .105: 195665 bytes, SHA256 `a334532aa2787ed9b3050c287be09f4c961131ba855bd6d1d7a5ef6f6c063e72`.

Host metadata originals: `bc250work/evidence/lora-resume/*-evidence-final.tar.gz`.
Archives include source/config, full proof transcripts, checkpoint markers,
kernel logs and acknowledged guard records; they exclude datasets, token IDs,
model/optimizer tensors and binary RNG payloads. The verifier checks integrity,
transcript equality and failure classifications, not rerunning GPU math.

Exact commands/configs and controller/worker identities are in phase
`running.json`, `final.json`, `replay-rank*.json` and controller logs. Guardians:
.104 R1seed12101/R2seed13638/R2resume15398/R3resume3944;
.105 R1seed9035/R2seed12975/R2resume17836/R3resume7663. No jobs, containers or
armed guards remain. Final boot IDs are
`7b7d244f-d349-420b-a2f5-ff7a68b5c7bb` (.104r2) and
`153daab4-ba4d-4141-8bae-48be45a8b2af` (.105r3).

**Next: D24**, BF16 FSDP working parameters/all-gather buffers with FP32 masters
and reductions. First carry the Docker-aware preflight into its owning guardian.
L02 does not widen D23's frozen-parameter or package-version guards.
