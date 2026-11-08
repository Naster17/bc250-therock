# D26: tied causal-LM Trainer training with classic FSDP

Completed 2026-11-08. Implementation/verifier `8e04668d7`.
[Verified results](verified-results.json),
[usage](../../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-causal-lm-and-tied-weights-d26).

The existing source-pinned BC250FSDPTrainer now has bounded qualification for an
owned eager GPT2 causal LM with tied embedding/output-head weights, under FP32,
BF16 working parameters and FP16 working parameters/native ShardedGradScaler.
No production Torch/ROCm defect was reproduced; native providers, public
launchers and policy helper are unchanged. The code adds a GPT2 fixture and
parameterizes the retained D25 reference/restart harness instead of copying its
CPU-gradient, clipping, AdamW, scaler and checkpoint logic. A BERT FP32 control
passes through the shared harness's original default. Existing D25 host tools
remain byte-preserved; new work is additive under `tools/distributed-d26`.

## Actual GPU and CPU checks

Two physical native gfx1013 devices, world2/RCCL, classic FULL_SHARD/original
parameters, root plus two GPT2Block wraps. All 21,280 unique incoming parameters
are trainable FP32 masters; both parameter and AdamW shards own10,640 elements
per rank. Embedding and LM head stay in the SAME root wrapping unit and retain
the SAME Parameter object; canonical aliases are transformer.wte.weight and
lm_head.weight. Checks run at construction, wrapping, full-gradient inspection,
updates, checkpoint restore and replay. Finite seed phases observe18 alias checks;
finite replay9; overflow seed15/replay7, because skip windows bypass optimizer
reference callbacks. No duplicate optimizer update or untied copy is introduced.

Owned data has28 examples/right padding/prompt masks/dropout0/cacheFalse/eager
attention/workers0/sequential/micro2/accum2. Global windows are8/8/8/final4.
D16 CausalLabelCollator masks position0 and padding before native Trainer counts;
causal_token_loss computes shifted token-summed FP32 CE divided by GLOBAL count.
Native counts49/57/57/21 differ from raw unshifted51/60/59/23. One actual rank0
micro has zero targets but a positive global denominator: finite differentiable
GPU loss0 and subsequent global-gradient references pass. The seven D16 CPU
analytic/ownership/refusal patterns and eight FSDP policy patterns are retained.

Every finite seed window compares global loss and ALL raw gradients with CPU
FP64 at actual FP32 or rounded BF16/FP16 weights. AMP also replays each rank's
real microbatch ordering with CPU autocast/low-precision accumulation and FP32
reduction/unscale. These use the same pinned HF graph with different CPU
arithmetic, not an independent Transformer implementation. Existing D15/D25
bounds are unchanged: BF16 relativeL2<=.02 CPU-AMP/.03 FP64; FP16<=.004/.006;
every parameter also has its norm-scaled bound plus3e-6*sqrt(elements). Maxima:

| Working precision | CPU AMP relative gradient error | CPU FP64 relative gradient error |
| --- | ---: | ---: |
| BF16 | .006585226 | .006604265 |
| FP16, finite | .000825151 | .000917906 |
| FP16, overflow fixture's finite windows | .000825151 | .000852122 |

FP32 loss atol/rtol3e-6/5e-5 and ALL gradients3e-6/5e-4 pass. AMP loss bounds
remain BF16.003/.003 and FP16.001/.001. Native root global norm/clipping at.05
matches CPU double actual-gradient predictions (norm1e-7/1e-5, every clipped
entry1e-8/5e-5). CPU double AdamW restores original unrounded masters and consumes
predicted clipped GPU gradients, isolating optimizer arithmetic from admitted AMP
approximation. ALL parameters, exp_avg, exp_avg_sq and counters pass the retained
D21/D22 bounds. Actual Conv1D and LM-head hooks plus training-only all-gather and
reduce-scatter observations verify working precision and FP32 synchronized
gradients/reductions/master/AdamW state. Padding-independent sharding/gather dtype
observations are not total NIC traffic, peak-memory or capacity measurements.

FP16 stress initial128/growth2 reaches512 after four finite updates. Injecting
rank1-owned Inf AFTER gradient reduction at window3 yields flags[False,True].
Native shard-aware unscale makes BOTH ranks skip AdamW and scheduler, preserve
parameters/moments/counters EXACTLY, and back off256→128; window4 executes update3.
Native production defaults65536/growth2000 are unchanged. Trainer global_step and
data/RNG still advance on skips; native aggregate train_loss is process-local.

## Exact fresh-process restart

Native checkpoint2 and rank-specific local Trainer/scheduler/scaler/four-RNG files
are preserved. Existing bounded offline DCP export/merge/install utility completes
six model/optimizer files on BOTH independent filesystems:653339B for each finite
precision,653347B for overflow. Original local checkpoint hashes remain unchanged;
only missing peer payloads are added. No private application/checkpoint is read.

Four new constructor/container/UUID/start-tick sessions per host use constructor
seed1773 instead of773 and FP16 initial scale8 instead of128. Native restore
matches ALL master shards, AdamW moments/counters, scheduler, scaler256 where
applicable and four RNG states EXACTLY. Remaining windows3/4 reproduce ALL losses,
gradients, clipping/norms, parameters/optimizer/scaler/scheduler, draws, data rows,
shifted counts, transcript digests and final state EXACTLY for FP32/BF16/FP16 AND
overflow replay. Tied weights remain shared throughout. This is fixed-world,
complete-optimizer-boundary replay; no stochastic/shuffled/multiworker/partial-
window/elastic/changed-policy or cross-native-version bitwise guarantee.

## Evidence and limits

**232 GPU groups /18 acknowledged guards**,15 CPU contract patterns observed256
times,62 local regressions, Black/critical Ruff+F401/diff checks pass. All GPU
phases pass on the first staged revision; no numerical tolerance is relaxed.
Initial archive verifier incorrectly demanded at least8 alias observations in
all phases; correct overflow replay has7 because the skipped update bypasses
callbacks. The verifier now checks exact18/15/9/7 counts by phase. GPU data was
unchanged and not rerun for this bookkeeping fix. Initial verifier/failure log
and r1 archives remain; r2 exports include the corrected verifier and assert the
original handoff bytes are unchanged. No full upstream CI.

Final tools on BOTH hosts: `/home/nik/bc250work/tools/distributed-d26`.
GPU evidence: `evidence/distributed-d26-r1/{fp32,bf16,fp16,overflow}-{seed,resume}`
and bert-control-seed, each with exact command/PID/container/boot/worker/guard/
ACK/registry/kernel logs. Controllers1042733/3398/5066 and1053761/4659/7063 exit0;
ordered child commands/PIDs/exits in seeds-sequence.json and resumes-sequence.json.
Final metadata exports7095/10554 exit0. Local `kernel/artifacts/distributed-d26`
preserves both original exports, four complete payload unions, source overlay,
historical verifier and metadata archives (tensor/RNG payloads excluded):

- .104 r2:672202B SHA25661aed24e5000b4b5be3d98d69a5ed43369a5f4146e931d96bbdcb8b4c4553037.
- .105 r2:568511B SHA256757620d5fee9ea29a49f97a2ad9f174371e43406b3b351443d918a0ca55845f1.

PLAN-01 exclusive cooperative board reservation +SH runtime locks are held
through each test.4CPUs/cpuset0–3/10GiB RAM/12GiB RAM+swap/512PIDs/256MiB shm,
600s recovery/460s child/15s termination retained. Final idle preflight passes,
no jobs/containers/armed guards, all kernel logs and D01 registries unchanged.
Bootsafcc7be4(.104r2)/8c396bd1(.105r3) unchanged. PLAN-02 selected MIOpen381eac05,
Torch a524, RCCL/D09, images/packages/defaults/boot fallbacks remain selected.
No native build/prefix change/reboot/download/app access/foreign stop. Historical
Python CPU faults remain distinct from fixed VM heap/D05 exit/D21 loading.

No speed, larger-model fit, convergence or universal causal-model claim. SDPA,
RoPE/GQA, generation/cache, packing/left padding, FSDP2/offload/compile, frozen/
PEFT and activation checkpointing are separate. **Next PLAN-04 / D27 (planned):**
inspect existing frozen/missing-gradient/optimizer guards, then qualify a small
explicit FP32 frozen-base/adapter FSDP fixture with CPU ALL-trainable-gradient/
AdamW/exact-restart checks before admitting that policy. Retain the current
frozen-parameter refusal until the new path proves its own contract.
