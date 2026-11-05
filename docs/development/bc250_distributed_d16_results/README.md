# D16: causal Transformer AMP and shifted-target normalization

Completed2026-11-05. Helper/contracts/fixturesce09f6c24, exportsad6454602. New opt-in
bc250_trainer_causal.py staged BOTH tools/distributed-d16; existing D12/D13/D14
normalization/finite/clipping helpers unchanged. Native daily/RCCL Torch a524980/
ROCm/D09 launcher/registry/images/profiles/kernels/frozen consumers unchanged.
No native build/install/prefix mutation/reboot/model download/user-app access.
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#causal-transformer-training-and-target-counts-d16).

## Actual causal graph and count contract

Owned GPT2LMHeadModel21,280 parameters, vocab97/positions32/embedding32/two layers/
four heads/inner64/sequence12, eager CAUSAL attention, learned positions, tied
embedding/LM head, Conv1D projections/GELU/LayerNorm/residuals, dropout0/cacheFalse.
Actual Transformers5.2.0/Accelerate1.12.0 Trainer BF16/FP16, not a substitute loop.
One native gfx1013 GPU/node/world2/RCCL, FP32 masters/logits/reductions, observed
Conv1D+Linear BF16/FP16 outputs. No RoPE/GQA/SDPA/generation-cache qualification.

Pinned Trainer counts labels.ne(-100) over the entire accumulation window and
both ranks. Causal CE uses logits[:,:-1] against labels[:,1:], so a nonignored
position0 must NOT inflate that denominator. New opt-in CausalLabelCollator wraps
a caller collator/default stacker, requires existing int64 labels matching inputs,
clones them, ignores position0 and attention-mask padding BEFORE Trainer counts.
Prompt ignore masks remain; caller labels/data are not mutated. Right-padding
binary masks only with at least two real tokens/sequence; left padding refused.
No labels are synthesized, packing/position-ID semantics inferred, or global
package default changed. Without a mask the caller owns padding-label masking.

New causal_token_loss requires this label contract and a positive supplied GLOBAL
window count. It sums FP32 CE for shifted logits/labels, divides by that count;
Trainer then compensates DDP averaging. model.accepts_loss_kwargs=False and the
D12 adapter prevent competing native-loss kwargs/double accumulation division.
A LOCAL all-ignored micro yields finite differentiable zero when global count>0.
A globally empty window/missing count or unmasked first label is refused.
This is an explicit training loss API, not a repair automatically applied to
ordinary Trainer/GPT2 calls or every native causal loss implementation.

28owned records, right padding5–12, prompt masks and two all-ignored examples on
rank0's first micro. Sequential workers0/per-device micro2/accumulation2: global8
first three attempts, final short global4. All rows and native supplied denominator
checked each attempt. Correct shifted global counts49,57,57,21 vs raw original
label counts51,60,59,23. Rank0 local active counts [0,16],[8,17],[8,16],[9]; peer
supplies the remainder. A raw denominator would shrink causal losses/gradients.
The zero-target local GPU micro is checked finite and exactly zero. Gradient
reductions0 nonfinal/85,120 FP32 bytes final; full graph gradients still execute.

## Numerical, clipping and optimizer references

Same declared D15 bounds, no relaxation. CPU native-autocast micro/rank/scale GPT2
and full-global CPU FP64 GPT2 at current GPU weights; both use manual shifted
active-token logsumexp minus selected logit loss, not GPU cross_entropy. Same
pinned HF graph source, separate CPU arithmetic; not independent GPT2 graph code.
Whole-model gradient relativeL2 limits AMP.02/.004, FP64.03/.006 (BF16/FP16).
EVERY parameter norm error <=relative_limit*reference_norm +floor*sqrt(elements),
floor3e-6/3e-7 for near-zero cancellation. Loss atol/rtol.003/.001 for BF16/FP16.
Observed maxima across finite windows/ranks:

| Precision | vs CPU autocast | vs CPU FP64 |
| --- | --- | --- |
|BF16|0.00399075|0.00495313|
|FP16|0.000498595|0.000661749|

Clipping.05 ACTIVE: unscaled norms1.15832–2.31724, coefficients.0215774–.0431659.
CPU FP64 ALL-grad global L2/coefficient predicts EVERY clipped gradient; native
norm rtol1e-5/atol1e-7, ALL clipped gradients rtol5e-5/atol1e-8. CPU-double AdamW
uses expected clipped gradients and checks ALL parameters/moments under retained
D07/D14 bounds. AdamW lr.001/eps1e-5/decay.01/foreachFalse, StepLR(step2,gamma.8),
raw post-step and independent LR/Adam step counts retained. Seven CPU contract
patterns execute per worker (56 observations): ownership/padding/count, left-pad
refusal, analytic shifted CE/ALL-logit derivatives, all-ignored local zero loss/
derivatives, missing/empty global count refusals and unmasked position0 refusal.

## Native AMP and restart

Fixture-specific GradScalerKwargs(init_scale128,growth_interval2) via pinned
Trainer extension, unchanged native defaults. FP16 scale128/tracker1→256/tracker0
at checkpoint2; rank1 Inf after third final DDP backward is agreed BEFORE native
unscale through D14 helper. BOTH native scalers skip AdamW/StepLR with weights/
moments/scheduler unchanged, back off128/tracker0, then finite short final window
recovers128/tracker1. Actual FP16 updates3 vs Trainer attempts4/LR.0008, BF16
updates4/LR.00064. No default65536 first-update or arbitrary-scale qualification.

Native save_on_each_node complete checkpoint2 on BOTH independent filesystems,
including model/optimizer/scheduler/Trainer/RNG/scaler. Local source/config/rank/
file SHA/size admission; new containers/workers/different constructor seeds call
actual native train(resume_from_checkpoint). Initial ALL params/AdamW/scheduler/
scaler/global_step/four RNG exact. Attempts3/4 ALL losses/gradients/parameters/
optimizer/scheduler/scaler/RNG/probe draws/rows/finite flag/global+local target
counts EXACT with zero tolerances and digests:8resumed rank-windows. Nonfinite
masks and finite components exact per D14, not NaN-payload-bit identity. Fresh
CID/UUID/startticks prove process restart. Dropout0 and RNG probes do not qualify
stochastic model or shuffled/multiworker/general checkpoint recovery.

## Evidence and operation

96accepted check groups/8ACK guards/8exact resumed rank-windows/42localtests,
7CPUcontract patterns/56observations separately. First revision all passes;
Black/critical Ruff+unused-import/syntax/diff pass, no full upstream CI. Source
verifier binds helper/contracts/fixture/normalization/finite/clipping and compares
cross-host config/counts/kernel/guard/registry/replay evidence. D14 refusal tests
retained separately; not repeated or counted here.

SH runtime locks,4CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,512PIDs/256MiB shm,
600s guards/460s children. Clean agents/phase kernel logs/registry. Controllers
19820/19250 exit0; commands/PIDs/exits sequence.json. No jobs/containers/armed
guards remain, foreign workloads untouched. Boots dc9e9293(.104r2)/133b8dc2(.105r3)
unchanged, SSH/boot fallback/native controls preserved. Native tied lm_head
missing-key notices pass complete exact restored state. Native all_gather/barrier
warnings and D12 aggregate resume train_loss caveat remain. Owned zero-loss assert
also emits scalar-conversion warning; actual loss stays a Tensor for backward,
not a production-helper warning. Historical Python CPU fault/KFD/invalidvisibility
remain distinct from fixed VMheap/D05 agent exit; no new native defect/speed claim.

Both tools/distributed-d16 and evidence/distributed-d16-r1/{bf16,fp16}-{seed,resume}.
Normal seed/native/reference payload3,731,560B/host retained plus resumed outputs.
Metadata exports omit.pt/.pth/.bin/.pkl/.safetensors, keep source/reports/registry/
kernel/guard/logs. Local kernel/artifacts/distributed-d16; committed
[verified summary](verified-results.json).

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|248327|691a0246b4a54b796eb3b5c1b33722d82082b092ba3182d5809d4e0c82a2cd88|
|105-evidence-r1.tar.gz|225607|f0e326c5089071e1d9063df01e2b931361170a39c5e04a1edb7829c000a6bbb6|

Both docs/memory synchronized. Scope: small eager GPT2/dropout0/cacheFalse,
right-padded unpacked sequences/explicit prompt ignore labels/positive global
window, one GPU/node/fixed world/sequential workers0. No left-padding/packing/
RoPE/GQA/SDPA/Flash/cache/generation/FSDP/DeepSpeed/gradient checkpointing/
arbitrary architecture/optimizer/scale/eval/convergence/capacity/speedup/elastic/
general resume proof. NextD17: actual two-host Trainer evaluation/prediction and
explicit token-weighted metrics, unequal final batches, CPU references. Invalid-
visibility frontend/B07.13/larger capacity/release reconstruction separate.
