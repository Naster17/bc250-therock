# D23: actual Trainer FSDP BF16 computation with FP32 parameters and reductions

Completed 2026-11-05. Helper/policy/fixtures `c49d99b84`, export/verifier
`0ecfd6258`. [Verified evidence](verified-results.json).
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-bf16-compute-policy-d23).

Added explicit `fsdp_bf16_compute_policy()` and a narrowly admitted BF16 path in
`BC250FSDPTrainer`. It combines native GPU BF16 autocast with FSDP parameter,
reduction and buffer dtypes ALL FP32; forward-input casts and low-precision kept
gradients disabled. Master weights must already be trainable FP32. No scaler.
Caller must set bf16=True AND this explicit mixed_precision_policy BEFORE
Trainer creation; default/low-parameter/low-reduction/input-casting policies are
refused. The policy is checked again after Accelerator creation. This does not
silently convert a user's model, choose BF16 parameter communication, or lift the
FP16/scaler/FSDP2/offload/compile/activation-checkpointing/frozen-parameter guards.

This is **BF16 computation**, not BF16 parameter sharding. GPU Linear hooks
observe BFloat16 outputs and FP32 forward weights; gradients/master shards and
public reduce_scatter_single inputs/outputs are FP32. Model loss receives FP32
prepared logits. Gradient communication performs three native reductions at each
final accumulation micro, zero at nonfinal;361024 input bytes/12 calls over four
updates,180512/6 on replay. These exclude parameter all-gathers, controls,
checkpoint traffic and physical wire overhead. No communication/parameter-byte
reduction, speedup, peak-memory saving or larger-model-fit claim.

## CPU references, clipping, AdamW and exact replay

Same owned BertMLM22561/vocab97/hidden32/two layers/four heads/inner64/positions32,
eager/dropout0/cacheFalse; root+two BertLayer FSDP units, classic FULL_SHARD/
use_orig_params/world2/one gfx1013 GPU per host/RCCL. Owned28 records with unequal
masked target counts/right-padding5–12; workers0/sequential/per-device micro2/
accum2 gives global8/8/8/final4. D21 single loss normalization/global target count
and repaired sharded AdamW restart, D22 native global clipping/collective finite
checks retained. Parameter/exp_avg elements11282(.104)/11279(.105), FP32.

At each of four seed updates, CPU BF16-autocast references replay BOTH ranks'
actual microbatch grouping/FP32-master accumulation at current GPU weights using
manual logsumexp CE; a separate CPU FP64 model sees the global window. Same HF
graph, different CPU arithmetic, not an independently implemented Transformer.
D15 BF16 limits declared/reused unchanged: loss atol/rtol.003/.003 against both
references; whole-gradient relative L2<=.02 versus CPU AMP/<=.03 versus FP64,
EVERY parameter's error norm <=limit*reference_norm+3e-6*sqrt(elements). That floor
handles near-zero key-bias cancellation, not an exception for large gradients.

Maximum observed whole-gradient relative errors: CPU AMP **.0006256018**,
CPU FP64 **.0048790555**. All individual parameter bounds pass. FP32 control uses
D21 strict loss3e-6/5e-5 and all-gradient3e-6/5e-4, not BF16 bounds.

CPU FP64 actual-pregrad global L2 norm predicts native root FSDP norm at
1e-7/1e-5; every clipped full gradient at1e-8/5e-5. Cap.05 active throughout:
norms1.46963418–2.26597190/coefs.03402205–.02206558; independent BF16 cap100 control
coefficient EXACT1. CPU-double AdamW from predicted clipped gradients checks ALL
parameters4e-6/5e-5/moments3e-7/5e-4 and3e-9/1e-3/counters exact. No tolerance
relaxation. This isolates optimizer accuracy from admitted BF16 approximation;
not an unrounded FP64 optimization trajectory or convergence proof.

Native checkpoint2/sharded optimizer reader +unchanged D21 offline payload
completion: six files864357 bytes; original hashes/local Trainer/scheduler/rank
RNG unchanged. Fresh CIDs/UUID/startticks/constructor seed1773, initial ALL shards/
AdamW/scheduler/four RNG EXACT, then windows3/4 ALL raw+clipped-gradient-derived
state/norms/coefficients/losses/weights/moments/scheduler/RNG/probes/rows/digests
EXACT versus uninterrupted BF16 run. No stochastic/shuffle/multiworker/elastic/
partial-window/peer-loss proof. D12 aggregate native train_loss caveat remains.

## Controls, provenance and handoff

Five phases/node: BF16 seed, BF16 resume, BF16 inactive clipping, one-step FP32
compatibility, rank1 local NaN refusal after valid BF16 backward. NaN raw flags
[False,True], both stop before clipping/AdamW; zero actual updates and unchanged
parameters/optimizer/scheduler. Transient gradients/RNG not rolled back; discard
aborted session and reload. D22 late-callback/collective-order limitations remain.

64 accepted GPU groups/10 acknowledged guards, four CPU clipping/policy patterns
observed40 times;42 local regressions/Black/critical Ruff+F401/syntax/diff pass.
First GPU revisions all pass; no failed GPU phase, no full upstream CI.
Source/config/helper/guard/kernel/registry/agent checks, native checkpoint hashes,
replay/dtype/CPU bounds and cross-rank agreement verified. Archive verifier does
not reexecute GPU arithmetic. Native FSDP/DCP/ShardedTensor warnings stay visible.

Both `tools/distributed-d23` contain final helper/factory and fixtures. Both
`evidence/distributed-d23-r1/{bf16-seed,bf16-resume,inactive-seed,fp32-control-seed,
nan-refusal}` plus controller logs/controls-sequence.json retain commands/PIDs/
CIDs/rank.json/final.json/ACK/kernel/source metadata. Seed16859/9394, controls
17410/9898, resume18750/11130 exit0; phase controls journal has exact worker PIDs.
Checkpoints2/4/reference tensors and payload exports preserved on boards; metadata
archives omit .pt/.pth/.bin/.distcp but inventory/hash them. Workstation keeps
payload exports/union and raw archives. Old D21/D22 tools/evidence untouched.

No native build/install/prefix mutation/reboot/model download/user application
inspection or foreign workload stop. Native daily/RCCL Torch a524980/ROCm/D09
launcher/consumers/profiles/images/registries/kernels/defaults unchanged. Same
SH runtime locks/4CPUs/cpuset0–3/10GiB RAM/12GiB memory+swap/512PIDs/256MiB shm/
600s recovery/460s child. Boots c1199fca(.104r2)/bad537ed(.105r3) unchanged.
Historical Python CPU fault/KFD/invalid visibility remain separate from fixed VM
heap/D05 exit/D21 optimizer-load repair. No jobs/containers/armed guards remain;
both host docs and agent-memory snapshots synced at completion.

Workstation `kernel/artifacts/distributed-d23` metadata archives:

- .104:368005 bytes, SHA256 `92e247e52028f1002480c27e9d67cc0fc77357a095b19945d6ee547df40c638e`.
- .105:322913 bytes, SHA256 `f1839d770ecfd1232d123020f6c9262c49b4625aa81bd6f31b2457c6c2fef51d`.

**Next D24 (planned):** audit/prototype BF16 FSDP forward/backward parameters and all-gather buffers
with FP32 master shards and reductions and round-aware CPU references, clipping/AdamW/fresh restart.
Keep D23 compute-only FP32-parameter policy as the qualified option until proof.
FP16 scaler/overflow policy, larger-model fit, general architectures and elastic
recovery remain separate priorities; dtype support does not imply speed/capacity.
