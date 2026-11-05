# D17: distributed Trainer evaluation/prediction and token-weighted metrics

Completed2026-11-05. Metric helper/contracts/fixturesf8f0a475a, exportscf7eb27c2.
New opt-in bc250_token_metrics.TokenWeightedMetrics staged BOTH tools/distributed-d17;
existing D12–D16 helpers unchanged. Native daily/RCCL Torch a524980/ROCm/D09
launcher/images/registry/profiles/kernels/frozen consumers unchanged. No native
build/install/prefix mutation/reboot/model download/user-app access.
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#distributed-evaluation-and-token-metrics-d17).

## Native orchestration and distinct metric semantics

Actual pinned Trainer.evaluate and Trainer.predict, one native gfx1013 GPU/node,
MULTI_GPU/world2/NCCL(RCCL). Owned small BertForMaskedLM22,561 and GPT2LMHeadModel
21,280, eager attention/dropout0/cacheFalse, existing D15/D16 graph configurations.
Each runs FP32, BF16 and FP16: six cases/two nodes. GPU parameter placement and
native Linear/Conv1D activation dtypes checked; prepared logits FP32. This is
forward/evaluation qualification, not new backward/scaler/optimizer proof.

Native prediction_step computes the supplied loss with each distributed batch's
global valid count; evaluation_loop repeats those local losses per sample, gathers,
then averages the loss array. That native eval_loss is NOT the whole-dataset token
NLL when token counts/batch layouts differ. D17 retains native eval_loss and adds
explicitly named metrics; no silent semantic replacement or native package patch.

TokenWeightedMetrics consumes native gathered FULL logits/int64 labels AFTER
native tail de-duplication. Masked mode compares matching positions; causal mode
shifts logits[:,:-1]/labels[:,1:] exactly once. Ignore targets-100. Sum CPU FP64 CE
and correct argmax counts over ALL active tokens, then divide ONCE by the dataset
active count. Returns token_nll, exp(token_nll) token_perplexity, token_accuracy,
active_tokens and evaluated_examples. Stateless across calls; chunked FP64 CE
work (fixture chunk7) matches independent manual NumPy64 logsumexp references.
No targets/full-logit wrong shape/invalid active logits or label IDs are refused.
No batch_eval_metrics/argmax-only/generation metric API is provided.

These are random owned fixtures, not pretrained quality or convergence results:

| Case | Native eval_loss | Explicit token_nll | Active targets / examples |
| --- | --- | --- | --- |
|Bert FP32|4.595089912|4.568608723|126 /19|
|Bert BF16|4.595186710|4.568724066|126 /19|
|Bert FP16|4.595079899|4.568599818|126 /19|
|GPT2 FP32|4.609643459|4.586088138|120 /19|
|GPT2 BF16|4.609602451|4.586088898|120 /19|
|GPT2 FP16|4.609643936|4.586089732|120 /19|

Native-vs-explicit gap.02351–.02648 in this deliberately unequal fixture establishes
a weighting distinction, not a GPU arithmetic fault. For token-weighted best-model
selection use metric_for_best_model='token_nll', greater_is_better=False; do not
assume a custom metric name automatically selects minimization.

## Dynamic padding, tail gathering and references

19records NOT divisible by world2*localbatch2. Variable actual sequence lengths5–12
are padded per local batch by an owned collator; labels are padded-100 and logits
padded natively across ranks/batches. Right-padding masks, prompt ignore labels,
two fully ignored examples on rank0's first batch; global batch still supervised.
CausalLabelCollator masks first label before Trainer count; Bert labels remain
unshifted. Sequential workers0, per-device eval batch2, eval_accumulation_steps2,
batch_eval_metrics=False/eval_do_concat_batches=True.

Native gather_for_metrics must remove the repeated tail row from even-batch
padding. Returned logits EXACT shape[19,12,97], labels[19,12] equal expected arrays
in global order, token counts126(masked)/120(causal), never20examples. EVERY real
input position's full-vocabulary GPU logits compared with CPU FP64 model forward:
FP32 atol3e-6/rtol5e-5; BF16.003/.003; FP16.001/.001. Same HF graph, different CPU
arithmetic; no independently implemented Transformer claim. Padded logits are not
CPU numerical comparisons because native gather may use sentinel-100 there.

All explicit metrics equal manual NumPy64 calculations on gathered logits within
1e-11. Explicit token_nll agrees with full-dataset CPU-FP64 model within the same
precision bounds. Token accuracy is compared with the ACTUAL gathered logits,
not assumed identical to FP64 argmax at near-ties. evaluate and labeled predict
metrics agree, both ranks return identical global metrics/prediction digest.

Label-free predict bypasses the training collator's required-label contract and
uses the base padding collator. It returns no labels/loss/token metrics and EXACT
same full logits as labeled predict, including native padding sentinels. No model
parameters/state-dict values or gradients change; all grads remain None. No
optimizer constructed or updated. Evaluation mode itself is an intentional model
mode change; no whole-process RNG/state rollback claim.

Five CPU contract patterns/60observations across12workers: masked/manual metrics,
causal/manual metrics, chunk1 vs4096, stateless repeated calls, empty-target and
argmax-only refusals (some checks grouped). These are separate from42local tests.

## Evidence, limits and next action

96accepted groups/12ACK GPU guards/5CPUpatterns*12workers=60observations/42local
regressions. First revision all phases pass. Black/critical Ruff+unused-import/
syntax/diff pass; full upstream CI not run. Archive verifier checks helper/fixture/
selected Trainer/native identity, raw rank/final records, kernel log equality,
guard nonces and cross-node metrics/counts/prediction digests; it does not re-run
GPU arithmetic from the archives. No persistent prediction/checkpoint payload was
created; deterministic fixture source and result digests/CPU comparisons retained.

SH runtime locks,4CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,512PIDs/256MiB shm,
600s guards/460s children. Clean agents/phase kernel logs/registries; controllers
25857/25113 exit0, exact commands/PIDs/exits sequence.json. No jobs/containers/
armed guards remain; foreign workloads untouched. Bootsdc9e9293(.104r2)/
133b8dc2(.105r3) unchanged; boot/SSH/native controls preserved. Native all_gather/
barrier warnings retained; historical Python CPU fault/KFD/invalid-visibility
failure distinct from already fixed VMheap/D05 agent exit. No native fault found.
No throughput, peak-memory, large-model fit or performance gain measured here.

Both tools/distributed-d17 and evidence/distributed-d17-r1/
{bert,gpt2}-{fp32,bf16,fp16}-eval. Workstation kernel/artifacts/distributed-d17,
committed [verified summary](verified-results.json) and raw metadata/source logs:

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|327082|2a49a451cc2816746e3e11d2c402ee51804bcc25216483eee4bd6f306b025636|
|105-evidence-r1.tar.gz|293063|9485e90a5f46d15e5311baf40266044912770389ef85a2ea72e4225484813f02|

Both docs/memory synchronized. Full logits are gathered/stored by Trainer. Chunking
only bounds temporary FP64 metric work, not dataset-wide logit storage/other
copies. Thus this is NOT memory-efficient large-vocabulary evaluation qualification.
No streaming/batch metrics, generated-text quality, IterableDataset, world/rank
changes, shuffled/multiworker, arbitrary preprocessing/model/backend/eval+train
resume or elastic/peer-loss claim. NextD18: compact per-example token statistics
before cross-host gather, CPU-reference parity, bounded payload/memory/timing
comparison. Retain native/default/recovery settings; other roadmap gaps remain.
