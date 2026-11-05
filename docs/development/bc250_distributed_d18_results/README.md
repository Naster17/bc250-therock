# D18: compact distributed token evaluation

Completed 2026-11-05. Helper/contracts/fixtures `9a31ac663`, export/verifier
`6bc35b079`. [Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#compact-distributed-evaluation-statistics-d18).
[Machine-readable observations](verified-results.json).

`CompactTokenMetrics` is an opt-in source helper, staged on both hosts under
`/home/nik/bc250work/tools/distributed-d18`. It supplies native Trainer's
`preprocess_logits_for_metrics` and `compute_metrics`. Before prediction gather,
it replaces each example's full vocabulary logits with three FP64 values:
summed NLL, active target count and correct argmax count. GPU-side FP64 CE is
chunked per example; causal slicing avoids a whole-batch flattened logit copy.
Native per-example tail de-duplication happens before final sums/division.

This is labeled evaluation only. Disable both callbacks to obtain full logits
or label-free predictions. Native Trainer loops, eval_loss semantics, D12–D17
helpers, daily/RCCL Torch a524980, ROCm, D09 launcher, consumer packages, images,
registries, profiles and kernels are unchanged. No native build/prefix mutation,
model download, application access or development reboot was issued.

## Correctness evidence

Actual native Trainer predict in ABBA order (full/compact/compact/full), compact
evaluate, then explicit opt-out to full-logit label-free predict ran on two
physical gfx1013 GPUs over RCCL. Six cases: eager BertForMaskedLM22561 and
GPT2LMHeadModel21280 in FP32/BF16/FP16, vocabulary97, dropout0/cacheFalse.
Owned19-example dataset has actual lengths5–12, right padding, unequal target
counts, prompt ignore masks and two completely ignored first examples. Native
world2/evalbatch2/workers0/accumulation-to-CPU2/concatenation settings retained.
Exact19 returned rows and expected labels verify tail de-duplication and order.
Counts126 masked/120 causal; individual zero-target rows contribute zero.

All compact metrics agree with full-logit metrics and independent NumPy64
logsumexp/argmax calculations within atol/rtol1e-11. Full GPU logits at all real
input positions match CPU FP64 model forward at D17 bounds: FP32 atol3e-6/
rtol5e-5, BF16 .003/.003, FP16 .001/.001. CPU reference uses the same pinned HF
graph with different CPU arithmetic, not an independent model implementation.
Evaluate/predict metrics agree; native loss is EXACTLY unchanged across modes.
Both ranks agree. Opt-out recovers identical full logits with no labels/loss/
token metrics. All model state tensors unchanged, all gradients None; no
optimizer constructed. Native eval-mode changes are intentional, not RNG rollback.

108 GPU check groups /12 acknowledged guards; four CPU contract patterns in
12 workers (48 observations), including masked/causal chunk1/3/4096 manual/full
parity, zero rows/stateless calls, per-example tail slicing and seven invalid
input refusals. GPU metric chunk7 exercises multi-chunk rows. Black, critical
Ruff+F401, syntax, diff checks and42 local regressions pass. No full upstream CI.
Source hashes, selected native identities, configs, guards/ACKs, clean agent exit,
unchanged per-phase kernel logs/registries, cross-rank metrics/full prediction
digests, and archives were verified. Archive verification does not rerun GPU math.

## Payload, memory and timing

Returned prediction array per rank: **88,464 →456 bytes (194x smaller)**.
Native gathered labels still use1,824 bytes. Actual prediction-gather INPUT
payload summed over five batches per rank: **36,472 →240 bytes (~152x smaller)**.
Including gathered losses and labels, local gather inputs total37,264 →1,032
bytes (~36.1x smaller). These are measured tensor bytes, not physical Ethernet
wire traffic or total distributed communication: native padding/collectives and
protocol overhead remain. Full local per-batch model logits still exist.

Peak LIVE Torch allocator bytes and elapsed milliseconds below are bounded tiny-
fixture observations, not total GPU memory, host RSS, capacity or training speed.
Allocator reset occurs before each prediction; labels/loss gathering is native.
The fixture keeps reference predictions on CPU to verify exact parity, so no
host peak-memory saving is inferred. Each ABBA entry is one run, not a statistical
benchmark. Only full mode is warmed before ABBA: FIRST compact use includes
cold work and costs211–231ms; the second compact call costs43.5–49.3ms. Full
runs are mostly38.2–44.6ms, with one90.5ms outlier. Compact did NOT demonstrate a
latency improvement here; per-row/chunk launches, scalar validation synchronizes
and FP64 cost remain optimization targets. Default chunk4096 is CPU-contract
checked, not separately GPU-performance-qualified.

| Case | Host | Full / compact / compact / full ms | Full → compact peak allocator bytes |
| --- | --- | --- | --- |
| bert-fp32-eval | .104 | 40.53 / 231.03 / 45.11 / 38.98 | 305152 → 264704 |
| bert-fp32-eval | .105 | 40.40 / 229.90 / 45.15 / 38.91 | 304640 → 268288 |
| bert-bf16-eval | .104 | 44.55 / 214.42 / 48.76 / 42.94 | 310272 → 297984 |
| bert-bf16-eval | .105 | 44.54 / 214.42 / 48.65 / 42.92 | 314880 → 302592 |
| bert-fp16-eval | .104 | 43.93 / 222.58 / 49.26 / 90.53 | 310272 → 297984 |
| bert-fp16-eval | .105 | 44.04 / 222.59 / 49.26 / 43.81 | 314880 → 302592 |
| gpt2-fp32-eval | .104 | 40.18 / 210.96 / 43.64 / 38.20 | 295424 → 266240 |
| gpt2-fp32-eval | .105 | 40.16 / 210.88 / 43.52 / 38.21 | 294912 → 274432 |
| gpt2-bf16-eval | .104 | 42.57 / 215.48 / 47.85 / 41.90 | 303104 → 290816 |
| gpt2-bf16-eval | .105 | 42.49 / 214.40 / 47.80 / 41.84 | 309248 → 296960 |
| gpt2-fp16-eval | .104 | 43.57 / 211.98 / 47.20 / 41.45 | 303104 → 290816 |
| gpt2-fp16-eval | .105 | 43.54 / 211.97 / 47.12 / 41.40 | 309248 → 296960 |

No large-vocabulary/long-dataset/larger-model-fit or universal speed/memory claim.
Chunking bounds temporary metric work, not model logits, native sequence padding,
dataset labels or all process memory. No streaming/batch_eval_metrics, arbitrary
output tuples, generation, IterableDataset, multiworker, elastic recovery,
eval+train restart or new backward qualification. Local invalid input refusal is
not a collective asymmetric-fault recovery proof.

## Reproduction and retained artifacts

On both hosts, as root, with boards idle, the owned controller was:

```sh
python3 /home/nik/bc250work/tools/distributed-d18/run_distributed_d18_on_host.py
```

It deliberately refuses an existing evidence directory. Existing results must be
preserved; do not rerun over them. Controller logs:
`/home/nik/bc250work/evidence/distributed-d18-controller.log`.
Per-phase exact commands/PIDs/exits are in
`evidence/distributed-d18-r1/sequence.json`; phases
`{bert,gpt2}-{fp32,bf16,fp16}-eval` contain rank.json/gpu.log/final.json and guards.
Controllers30382(.104)/29377(.105) exited0. Six sequential guards per host,
SH runtime locks,4CPUs/cpuset0–3/10GiB RAM/12GiB memory+swap/512PIDs/256MiB shm;
600s recovery guards/460s child deadlines. No foreign workload was stopped.

Qualification boots: dc9e9293-451c-4cd0-8839-7c902c8509ad(.104r2),
133b8dc2-62a1-417b-8acb-fc44aeda6eec(.105r3). Both boards returned with new boot IDs
AFTER tests, guard ACK/export/verification and the chat interruption; no GPU
qualification is claimed on the later boots. Documentation synchronization uses
new boots d83af4e1-e3ef-4cc8-ade9-371328a22f4b(.104),
bad537ed-ea08-49e6-ad44-40a9bc1d6914(.105). No test jobs/containers/armed guards
remain. Native all_gather/barrier warnings remain visible. Historical Python CPU
fault/KFD/invalid visibility remain separate from fixed VM heap and D05 agent exit.

Workstation archives under `kernel/artifacts/distributed-d18`:

- .104: 344806 bytes, SHA256 `76691515b8b76a925ac7009466d3443b25640a9ecbd5d79b80f5c778d48a03bf`.
- .105: 311154 bytes, SHA256 `307692ad8da565a706edc16c09800bc5307b6b8a17d87b42f400d38771dcba2b`.

Sources, raw logs, summaries and all historical evidence are preserved. No
persistent prediction/checkpoint tensor payload was created. Both host docs and
agent-memory snapshots are synchronized at handoff.

**Next D19 (planned):** reduce compact metric launch/synchronization overhead,
retain exact count/tail/CPU contracts, and test a bounded larger-vocabulary
payload/memory/latency crossover. Keep opt-in behavior and native defaults.
