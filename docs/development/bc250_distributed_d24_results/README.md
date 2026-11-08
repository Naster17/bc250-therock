# D24: BF16 FSDP working parameters with FP32 master shards

Completed 2026-11-08. Implementation and verifier `dfe347376`.
[Verified evidence](verified-results.json),
[usage](../../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-bf16-working-parameters-d24).

Added explicit `fsdp_bf16_parameter_policy()` to the pinned classic
`BC250FSDPTrainer`. Native FSDP handles BF16 forward/backward parameters and
parameter all-gathers, FP32 master shards/reductions/buffers, no kept low-precision
gradients/input casts/scaler. Incoming parameters must remain trainable FP32.
D23 compute-only factory and old staged helpers remain available; FP16, frozen
parameters, other package pairs, FSDP2, offload, compilation and activation
checkpointing remain refused. No native Torch/ROCm rebuild or prefix change.

The D24 guardian checks default-local Docker before and after, in addition to
Podman, GPU file descriptors, watchdog, memory and the existing shared runtime
lock. An unknown Docker state refuses qualification; unrelated workloads are
never stopped. A post-test query failure marks the guard failed while allowing
recovery acknowledgment after its own workers exit. Custom/rootless daemons are
not exhaustively enumerated; older guardians outside D24/L02 still need migration.

## Numerical qualification

Owned eager Bert MLM: 22,561 parameters, two layers/hidden32/heads4/vocab97,
dropout0, root plus two BertLayer FSDP units, FULL_SHARD/use_orig_params, one
native gfx1013 GPU per node with RCCL. Twenty-eight sequential records with
unequal MLM masks and right padding, workers0/micro2/accum2: global windows
8/8/8/4. Source-pinned Transformers5.2.0/Accelerate1.12.0, native Trainer loop,
D21 token denominator and optimizer-aware checkpoint loader, D22 native clipping.

GPU hooks see BF16 Linear weights and outputs; public all_gather_single observes
BF16 input/output during native training forward/backward, and reduce_scatter_single
observes FP32. Gradients after synchronization and master parameters are FP32;
unsynchronized local gradients may temporarily be BF16. Native BF16 alignment
changes owned parameter/AdamW element counts to **11,284 / 11,277**, totaling22,561;
the FP32 control remains11,282/11,279.

CPU references use the same HF graph with different CPU arithmetic, not an
independently implemented Transformer. CPU BF16 parameter/autocast references
replay each rank's actual microbatches and BF16 local gradient accumulation,
then FP32 averaging. CPU FP64 sees BF16-rounded working weights and the global
window. All losses and ALL parameter gradients pass unchanged D15 bounds:
loss atol/rtol.003/.003; whole-gradient relative error<=.02 CPU BF16/<=.03 FP64;
every parameter error<=bound*reference_norm+3e-6*sqrt(elements). Maximum observed
relative errors **.008212774 CPU BF16 / .005715182 FP64**. No bounds relaxed.

CPU FP64 norm predicts native root clipping at1e-7/1e-5 and ALL clipped gradients
at1e-8/5e-5. Cap.05 is active in every window; norms1.469559–2.265428. CPU-double
AdamW restores original FP32 masters before updating from predicted clipped GPU
gradients; ALL parameters/moments/counters pass D21/D22 bounds. This isolates
optimizer arithmetic from BF16 approximation, not an unrounded FP64 trajectory.
Inactive cap100 coefficient1 and strict FP32 compatibility pass. A rank1 NaN
causes BOTH ranks to refuse before clipping/AdamW, zero actual updates and exact
unchanged parameters/optimizer/scheduler. Transient gradients/RNG are not rolled
back; discard and reload after refusal.

## Communication and restart evidence

Training-only gather inputs per rank: BF16 seed35calls/277,592B, fresh replay
15calls/118,968B; BF16 one-window inactive control10calls/79,312B versus FP32
control10calls/158,608B. These include padding and exclude full-state/checkpoint
probes. BF16 elements use2B instead of4B; padding differs. Four-update reduction
input361,088B/12calls, replay180,544B/6calls, allFP32. No total wire traffic,
physical NIC counters, peak memory, speed or larger-model-fit claim.

Native checkpoint2 plus unchanged offline DCP exchange:6files/864,357B,
original local checkpoint hashes/Trainer/scheduler/rank RNG unchanged. Fresh
containers/UUIDs/start ticks/constructor seed1773 restore ALL master shards,
AdamW, scheduler and four RNG states EXACT before training. Windows3/4 replay
ALL raw/clipped-gradient-derived state, norms, losses, parameters, moments,
scheduler, RNG draws, rows and transcript digests EXACT on BOTH ranks. This
covers a completed optimizer boundary, same world/rank/settings, deterministic
sequential workers0; no elastic/stochastic/multiworker/partial-window proof.
Native aggregate train_loss remains a current-process statistic.

## Provenance and handoff

64 accepted GPU groups,10 acknowledged guards,6 CPU policy/clipping patterns
observed60times,58 local tests; Black/critical Ruff+F401/py_compile/diff pass.
All GPU phases passed, no kernel-log/registry changes, Docker/Podman idle before
and after. Initial archive verifier overcounted gathers by assuming a full final
window; corrected35/15counts reflect actual8/8/8/4data, without GPU changes.
Initial and corrected metadata archives retained. No full upstream CI.

Both nodes retain `tools/distributed-d24`, checkpoints2/4/reference tensors and
`evidence/distributed-d24-r1/{bf16-seed,bf16-resume,inactive-seed,
fp32-control-seed,nan-refusal}`. Each final.json has exact command/PID/container,
boot ID, kernel/registry checks and acknowledged recovery; controls-sequence.json
has ordered commands/PIDs/exits. Seed guardians5142/9798, resume5835/10445,
controls controllers6208/10823, all exit0. No remaining jobs/containers/armed guards.

Shared runtime locks,4CPUs/cpuset0–3/10GiB RAM/12GiB RAM+swap/512PIDs/256MiBshm,
600s recovery/460s child unchanged. Boots afcc7be4(.104r2)/8c396bd1(.105r3)
already active at entry, retained; no agent reboot, download, prefix install,
foreign stop or user application inspection. Torch daily/RCCL a524980 and D09,
kernels/fallbacks/packages/defaults unchanged. Historical CPU Python/KFD faults
remain separate from fixed VM heap/D05 exit/D21 optimizer loading.

Local `kernel/artifacts/distributed-d24` retains full metadata archives and
checkpoint-payload exports/union. Archives omit tensor/RNG payload binaries;
inventories and hashes retained, no private application/data content exported.

- .104:1,226,233B, SHA256 `885ebd47cf3622918729a73f29b56fc95f2e281202115161447bc425d6a3c083`.
- .105:1,180,601B, SHA256 `cfe5d2fb1382df55dbec53cfc604b267ea242b6e0a7a6a5787095131adc57173`.

**Next D25 (planned):** audit/prototype actual Trainer classic FSDP FP16 working
parameters with native sharded scaler, CPU references, global clipping, collective
overflow/skip and exact fresh restart. Keep FP16 refusal until that proof. Larger
capacity, causal/other architectures, frozen parameters and elastic recovery
remain separate priorities; no universal correctness or convergence claim.
