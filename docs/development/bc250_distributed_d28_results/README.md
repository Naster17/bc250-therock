# D28: actual PEFT LoRA FP32 with classic FSDP

Completed 2026-11-09. Implementation/verifier `e08c98d15`.
[Verified results](verified-results.json),
[usage](../../../../BC250_DISTRIBUTED_GUIDE.md#actual-peft-lora-fp32-fsdp-d28).

New explicit BC250PeftFSDPTrainer/peft_fp32_policy supports the pinned PEFT0.18.1
single default causal-LM LoRA recipe on top of D27's frozen FP32 contract.
Transformers5.2.0/Accelerate1.12.0 and nine additional critical PEFT/FSDP source
hashes are checked. All old helper namespaces, native providers and packages
remain unchanged. New additive tools/distributed-d28 are installed on BOTH hosts.
No native Torch/ROCm rebuild or global PEFT/Trainer monkey patch.

## Audited integration changes

Pinned Trainer calls update_fsdp_plugin_peft before native wrapping. PEFT's
public auto-wrap policy adds separate trainable LoRA leaf units. The actual
fixture has root+two GPT2Blocks+16 adapter Linear units (19 total), not D27's
three mixed units. We retain this behavior and verify actual topology, gathers,
gradients/updates and restart rather than assuming a module-name match is proof.

Pinned Trainer also requests adapter_only=True in FSDP save/load helpers. An
adapter-only artifact is not a full frozen-base restart, particularly with a
fresh differently seeded constructor. The new subclass's checkpoint methods
call public Accelerate FSDP model save/load with adapter_only=False, native
FSDP optimizer save and native scheduler state. The Trainer loop, RNG/scaler/
Trainer-state handling, D27 policy and D21 optimizer-aware DCP reader remain.
The full checkpoint restores ALL base and adapter shards without an external
base-weight reload. This is an explicit checkpoint contract, not an upstream
PEFT defect or a conversion of old adapter-only checkpoints.

Checkpoint metadata binds canonical partition/shapes, PEFT configuration and
source hashes, base-model configuration/attention selection, active adapter,
per-layer rank/alpha/scaling/dropout/merge/offload state. Only plain single
unmerged default causal LoRA, dropout0/biasnone, no DoRA/RSLoRA/saved modules or
other variants is admitted. Every trainable name must be a default LoRA A/B
weight. D27's FP32/positive-root-clipping/trainable-only AdamW/gradient guards
remain. Changed scaling or activation/recipe is refused before update.

Each completed checkpoint also gets a separate peft-adapter directory, produced
by public PeftModel.save_pretrained with canonical unwrapped adapter keys and
save_embedding_layers=False. It contains ordinary adapter_config.json and
adapter_model.safetensors, not optimizer/RNG/base weights. R1's new export kept
FSDP wrapper prefixes: public loading warned about missing adapter keys, and
strict parameter equality rejected BOTH guards at checkpoint2. The integration
now strips only the known FSDP wrapper prefix. R2 passes; R3 adds the final base-
recipe binding/live scaling refusal and reruns all selected phases. No tolerance
was relaxed or missing-key warning suppressed. Old user training was not tested
or implicated by this prototype defect.

## Actual mathematical and public-package checks

Owned eager GPT2, tied frozen embedding/head, eight actual PEFT Conv1D LoRA
projections, r4/alpha8/dropout0/fan_in_fan_outTrue/zero-initialized B.24,352 unique
parameters=21,280 frozen+3,072 trainable. Per-rank original shards12,176 each;
adapter and AdamW elements1,536 each. The full model is still sharded/gathered.
Actual PEFT automatic wrapping gives16 reductions/window and259 training-only
gathers over seven seed microbatches (111 over three replay microbatches).
Reduce-scatter input is12,288B/window. Counts/payloads reflect this tiny topology;
they are not total NIC, peak memory or speed comparisons against D27.

D26/D27 owned28records/right-padding/prompt masks/dropout0/eager/cacheFalse/
workers0/sequential/micro2/accum2 yields global8/8/8/final4 and shifted target
counts49/57/57/21. Actual local-zero-target micro and caller-label contracts pass.
ALL adapter gradients and global loss match CPU FP64 at unchanged FP32 bounds
(loss3e-6/5e-5, gradients3e-6/5e-4). ALL frozen parameter values remain exactly
unchanged. Native root clipping at.005 matches CPU norm/every-gradient predictions
(norm1e-7/1e-5, clipped entries1e-8/5e-5). CPU double AdamW consumes predicted
clipped GPU gradients; every adapter value/moment/counter passes D21/D22 bounds.
The CPU uses the same pinned HF+PEFT graph with different arithmetic, not an
independent Transformer. Actual GPU hooks and FP32 gather/reduce observations
verify native device execution; no AMP/matrix-instruction claim.

Checkpoint2 AND4 adapter exports (14,216B each safetensors file) load through
public PeftModel.from_pretrained on a CPU copy of the matching frozen base.
ALL base/adapter parameters and logits match exactly. CPU audits restore RNG
through fork_rng and assert all four RNG states unchanged. Replay checkpoint4
export also passes. Adapter artifacts require matching base weights; they are
not substitutes for complete training checkpoints.

Four rank1-only live mutations are collectively refused on BOTH nodes with zero
AdamW updates and unchanged parameters/optimizer/scheduler: missing trainable
gradient, forbidden frozen gradient, post-clip NaN, post-clip LoRA scaling change.
A private checkpoint copy changes only rank1's saved LoRA scaling recipe while
leaving the generic frozen-partition hash unchanged; BOTH refuse optimizer
restoration with zero updates. Original checkpoint/policy hashes are preserved.
Model shards may already have loaded on refusal; discard an aborted session.
The D27 native-adapter FP32 helper still passes a retained control in this new
namespace. Twenty-six causal/frozen/precision/PEFT CPU contract patterns are
observed408times across the accepted final guards.

## Full exact restart and retained evidence

Native checkpoint2 DCP union: six model/optimizer files480247B on both independent
filesystems. Existing bounded offline exchange adds missing peer files only;
local Trainer/scheduler/RNG/recipe files stay local and unchanged. Fresh seed1773
instead of773, new UUID/start ticks/containers, restores ALL frozen/adapter shards,
AdamW moments/counters, scheduler and four RNG states exactly. Windows3/4 replay
ALL losses, gradients, clip/norms, values/moments/counters, scheduler/RNG/data/
target counts and transcript/final-state digests exactly. Native train_loss
remains process-local. No partial-window/stochastic/multiworker/elastic or
changed-policy replay promise.

**116 groups/16 final acknowledged guards**,26 CPU patterns/408 observations,
62 local regressions/Black/critical Ruff+F401/diff checks pass. Two R1 rejected
export guards and two R2 intermediate passed guards retained; all20 acknowledged.
No full upstream CI, no speed/larger-fit/convergence/universal-model claim.

Exact commands/PIDs/CIDs/ACKs: evidence/distributed-d28-r3/frozen-{seed,resume},
frozen-{missing,grad,late,peft}-seed, retained-control-seed, frozen-policy-resume;
controls-sequence.json/resumes-sequence.json/controller logs. frozen-policy-seed
is a private copy, not an additional run. R1 11804/13899 exit1; R2 12431/14692
exit0; R3 seed13445/16280, controls14014/17060, resumes16159/19784 and exports
17340/21442 exit0. Historical sources/checkpoints/references/artifacts remain.
Local kernel/artifacts/distributed-d28 holds package audit/pins, overlay,
original exports, complete union and metadata (tensor/RNG/safetensors payloads
excluded):

- .104:816023B SHA256eedfa6b5bf43807495d11bafd9d10591beede394d4b98f949acec2f6ca3e64ac.
- .105:713531B SHA2561fbd0d22fdbe0b9982499e470477606b35f7ee214262a77167ec45f01b79b4c1.

PLAN-01 EX board reservation+SH runtime locks/4CPUs/cpuset0–3/10GiB RAM/12GiB
RAM+swap/512PIDs/256MiB shm/600s recovery/460s child/15s termination retained.
Kernel logs/registries unchanged, final idle/no jobs/containers/armed guards.
Entry boots2d86d523(.104r2)/e902e636(.105r3), already different from prior D27
snapshots; no agent reboot. PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/image/
packages/providers/defaults/fallbacks preserved. No build/prefix mutation/download/
app inspection/foreign stop. Historical Python CPU faults remain separate from
fixed VM heap/D05 exit/D21 loading.

PLAN-04 remains OPEN. Next D29 (planned): audit an explicit non-reentrant FP32
PEFT activation-checkpointing policy, qualify ALL gradients/updates/frozen values
and native restart, then compare live allocator costs within safe headroom.
Frozen AMP and checkpointed/other adapter variants remain refused until their
own qualification; real capacity and performance remain separate work.
