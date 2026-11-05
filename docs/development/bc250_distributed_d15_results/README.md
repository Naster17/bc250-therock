# D15: offline Transformer AMP Trainer, token CE and native restart

Completed2026-11-05. Fixturesdeed14f14, exports84039269e. Existing D12 normalization,
D13 finite callback and D14 coordinated clipping helper unchanged; copies staged
in tools/distributed-d15 on BOTH hosts. Native daily/RCCL Torch a524980/ROCm/D09
launcher/images/registry/profiles/kernels/frozen consumers unchanged. No native
build/install/prefix mutation/reboot/user-application access or model download.
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#offline-transformer-amp-training-d15).

## Actual Transformer graph and normalization

Tiny owned offline BertForMaskedLM22,561 parameters, vocab97/hidden32/two layers/
four heads/intermediate64/maximum positions32/length12, eager bidirectional
attention, zero hidden/attention dropout. This is an actual Transformers5.2.0
model, extending D13/D14's residual-MLP AMP qualification. Embedding and tied
MLM decoder weights, masked attention matmuls/softmax, LayerNorm, GELU, residuals
and masked CE participate in forward/backward. Every trainable parameter gradient
is checked. No arbitrary architecture/backend/model-size claim.

28owned records contain unequal right-padding lengths5–12, nonzero attention
masks and unequal numbers of active MLM targets. Pad input ID0, ignored targets
-100. All row IDs and both ranks' global partition are checked. Sequential workers0,
per-device micro2/accumulation2: global8 first three attempts, final short global4.
DDP sends90,244 FP32 gradient bytes on final micros and zero on nonfinal micros.
One native gfx1013 GPU/node/world2/NCCL(RCCL), observed native Linear outputs
BF16/FP16 with FP32 master/reduction. Prepared Accelerator converts model outputs
back to FP32; the fixture explicitly checks logits dtype at the loss hook.

Model.accepts_loss_kwargs=False and unchanged masked_token_loss hook explicitly
own UNshifted masked-LM CE: summed active-token CE divided by Trainer's global
active count over the entire accumulation window. Trainer compensates DDP averaging;
BC250ClippedTrainer removes the pinned package pair's second accumulation division.
Do not average per-micro losses when valid-token counts differ, or use this hook
as a shifted causal-LM loss. Actual Trainer/DDP/loader/AdamW/StepLR/scaler/native
save/resume are used, not a hand-written training loop substituted for Trainer.

## CPU references, clipping and AdamW

Two numerical references, declared bounds before first GPU run; no bounds relaxed:

- CPU native autocast with the same precision/actual micro/rank grouping and
  actual loss scale, with separate CPU kernels. Loss uses an explicit active-token
  logsumexp minus selected logit sum, not the GPU cross_entropy call. Accumulated
  gradients are unscaled and averaged across the simulated ranks.
- CPU FP64 full-global-batch Bert graph with the same current GPU weights and
  manual logsumexp CE, differentiated without autocast/loss scaling. This is a
  high-precision mathematical comparison, not an independent Transformer graph
  implementation. Both references use the same pinned HF graph source.

Global loss comparison atol/rtol.003(BF16),.001(FP16). Whole-model gradient relative
L2 limits CPU-AMP.02/.004 and FP64.03/.006 for BF16/FP16 respectively. EACH parameter
also obeys error_norm <=relative_limit*reference_norm +floor*sqrt(elements), floor
3e-6(BF16)/3e-7(FP16), allowing near-zero attention key-bias cancellation noise.
These floors are explicit; no pure relative proof for effectively zero gradients.
Maximum observed whole-model relative gradient errors across finite windows/ranks:

| Precision | vs CPU autocast | vs CPU FP64 |
| --- | --- | --- |
|BF16|0.0000582494|0.00487906|
|FP16|0.0000732985|0.000594423|

CPU FP64 norm and coefficient min(1,.05/(norm+1e-6)) independently predict EVERY
clipped gradient; native norm rtol1e-5/atol1e-7, ALL clipped gradients rtol5e-5/
atol1e-8. Clipping ACTIVE: norms1.46933–2.27917, coefficients.0219378–.0340292.
CPU-double AdamW fed these expected clipped gradients checks ALL parameters and
exp_avg/exp_avg_sq under retained D07 limits. AdamW lr.001/eps1e-5/decay.01/
foreachFalse, StepLR(step2,gamma.8); independent raw-hook/step/LR checks retained.
Reference comparisons occur at four finite BF16 windows and three finite FP16
windows per rank. No optimizer step occurs on FP16's deliberately invalid window.

## Scaler, skip and exact native restart

Fixture-specific FP16 GradScaler kwargs init_scale128/growth_interval2 are passed
through the pinned Trainer's _build_accelerator_args extension. Native defaults
and installed helpers are NOT changed. This bounded test explicitly selects a
modest initial scale; it does not qualify Bert's first-update behavior at default
65536. BF16 has no scaler. FP16 scale128/tracker1→256/tracker0 at checkpoint2;
rank1-only Inf AFTER final DDP backward at attempt3 is agreed BEFORE native unscale
through the unchanged D14 adapter. BOTH native scalers skip AdamW/StepLR, preserve
weights/moments/scheduler, back off128/tracker0; finite short attempt4 recovers at
128/tracker1. Native Trainer global_step4 counts attempts; actual FP16 AdamW/
StepLR count3/LR.0008, BF16 count4/LR.00064.

Native save_on_each_node checkpoint2 exists independently on each host, including
safetensors/optimizer/StepLR/Trainer/RNG and FP16 scaler files. Hash/size/source/
config/rank admission before actual train(resume_from_checkpoint) in entirely new
containers/processes with different constructor seeds. Native skip/restore matches
initial parameters/optimizer/scheduler/scaler/global_step/four RNG streams exactly.
Attempts3/4 then match ALL local loss/gradients/parameters/optimizer/scheduler/
scaler/RNG/probe draws/rows/finite flag exactly with zero tolerances and digests:
8resumed rank-windows, including native growth/skip/backoff/recovery. Exact
nonfinite masks +finite gradient components retained as in D14; NaN payload bits
are not compared. Fresh CID/UUID/startticks prove process restart. Dropout0;
explicit Python/NumPy/Torch CPU/CUDA draws are RNG probes, not stochastic layers.

## Evidence and boundaries

70accepted check groups/8ACK guards/8exact resumed rank-windows/42localtests.
First revision all phases pass. Black/critical Ruff+unused-import/syntax/diff pass;
full upstream CI not run. D14 refusal tests remain separate retained evidence,
not repeated or counted here. Native tied decoder missing-key notices occur on
resume; complete model/optimizer equality and replay pass. Native all_gather
warning/barrier warning and D12 aggregate resume train_loss caveat remain.

SH runtime locks,4CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,512PIDs/256MiB shm,
600s guards/460s child bounds; clean agents and unchanged phase kernel logs/
registries. Controllers15951/15570 exit0; commands/PIDs/exits sequence.json.
No jobs/containers/armed guards remain; foreign workloads untouched. Boots
 dc9e9293(.104r2)/133b8dc2(.105r3) unchanged, fallback/native controls preserved.
Historical Python CPU fault/KFD/invalid-visibility failure remains distinct from
fixed VMheap/D05 agent exit. No new native defect or speed/capacity claim.

Both tools/distributed-d15 and evidence/distributed-d15-r1/{bf16,fp16}-{seed,resume}.
Normal seed/native/reference payload4,126,664B/host retained plus resumed outputs.
Metadata exports omit.pt/.pth/.bin/.pkl/.safetensors data, retain sources/reports/
registry/kernel/guard/log diagnostics. Local kernel/artifacts/distributed-d15;
committed [verified summary](verified-results.json).

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|239556|0a07815d632fcc7f636235191d868affec2bba17dd88b400e25ab2eee14e952b|
|105-evidence-r1.tar.gz|217535|6a05eb85c2ae290ecfc9cb68b92d7eea7f46256e701a2774ddfbde1db21479ba|

Both docs/memory synchronized. Scope remains owned small bidirectional Bert/eager
MLM graph, dropout0, fixed world2/one GPU per node, sequential workers0 trusted
optimizer-boundary checkpoint. No causal LM/RoPE/GQA/cache/SDPA/FlashAttention/
FSDP/DeepSpeed/gradient checkpointing/arbitrary architecture/loss/optimizer/scale/
large capacity/convergence/speedup/stochastic-model/multiworker/elastic/general
resume claim. NextD16: owned offline causal Transformer AMP Trainer with SHIFTED
CE/global valid-target accounting, clipping/references/native restart. Evaluation
metrics, invalid-visibility frontend and release reconstruction remain separate.
