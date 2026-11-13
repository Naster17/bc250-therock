# D67 — distributed value benchmark: when do two BC-250 boards pay off?

Completed 2026-11-13. This batch consolidates the distributed performance
evidence (D05/D32/D35/D43/L01) into one decision-ready picture and adds a new
bounded measurement: **two-board vs one-board LoRA throughput scaling across
sequence lengths**, using the authorized lora-smoke integration with **owned
synthetic data** (no user data). Single-board reference: .104; two-board: .104 +
.105 over 1 GbE RCCL (Ring/Simple, Socket, eth0).

**Headline answer:** two boards buy **capacity**, not speed. Full-parameter
training that fits one board is 6.8–68× slower on two; a model that does not fit
one board trains only via two-board FSDP (1.024B demonstrated); LoRA fine-tuning
of a 2B base is the one workload class where two boards are faster per update —
**~1.94× at both D67 operating points** (1024 steady state, 4096 cold) and
+23% on L01's real long mixed-length data — but whole-job startup is paid on
both boards, so short jobs remain faster on one.

## Consolidated evidence (prior milestones, unchanged)

All numbers are owned-fixture measurements on the native gfx1013 release, matched
global batch where compared, 1 GbE eth0 on both boards.

| Workload | Mode | Median update | vs single | Peak live GPU/rank | Source |
| --- | --- | ---: | ---: | ---: | --- |
| 12.61M FP32 transformer | single | 23.55–27.22 ms | 1× | — | D05 |
| 12.61M FP32 transformer | 2-host DDP | 438.25–438.41 ms | ~16–19× slower | — | D05 |
| 12.61M FP32 transformer | 2-host FSDP | 651.55–651.70 ms | ~24–28× slower | — | D05 |
| 45.9M GPT2 FP32 | single (.104) | 247.96–248.06 ms | 1× | 940.7 MiB | D32 |
| 45.9M GPT2 FP32 | 2-host DDP | 1681.58–1682.03 ms | ~6.8× slower | 1114.5 MiB | D32 |
| 45.9M GPT2 FP32 | 2-host FSDP | 6885.37–6885.57 ms | ~27.8× slower | 768.4–770.9 MiB | D32 |
| 352.96M GPT2 FP32 | single | 265.54–266.56 ms | 1× | 5863.5 MiB | D35 |
| 352.96M GPT2 FP32 | 2-host FSDP | 17968.64–17973.42 ms | ~68× slower | 3184–3190 MiB (−45.6%) | D35 |
| 1.024B GPT2 FP32 | single | **impossible** | — | state floor 16.39 GB > host RAM | D43 |
| 1.024B GPT2 FP32 | 2-host FSDP | 53.36 s | capacity enabled | ~8586–8588 MiB | D43 |
| Qwen3.5-2B LoRA r128 FP16 (real data subset) | single | 122.89 s / 5 updates | 1× | 9.167 GiB | L01 |
| Qwen3.5-2B LoRA r128 FP16 (real data subset) | 2-host DDP | 99.86 s / 5 updates | **1.231× faster** | 9.665 / 7.493 GiB | L01 |

Reading the table:

- **Full-parameter, fits one board:** never faster on two. The gradient/parameter
  traffic (e.g. D32 DDP 183.5 MB, FSDP 707.2 MB gather + 183.5 MB reduce per
  rank/update) crosses a 1 GbE link every update; compute per update is too small
  to hide it. FSDP's value here is memory headroom (−45.6% live GPU at 352.96M),
  not throughput.
- **Full-parameter, does not fit one board:** two-board FSDP is the only option.
  D43 ran eight 1.024B FP32 updates (53.36 s median) where the single-board
  all-resident parameter/gradient/optimizer floor (16,391,536,640 B) exceeds
  either host's RAM. Slow, but possible; no single-board alternative exists.
- **LoRA (frozen base, small trainable payload):** the base forward/backward
  compute dominates while only the adapter gradients synchronize (L01 rank-128:
  134,553,600 trainable FP32 params ≈ 538 MB/update). Two boards split the
  compute and pay one gradient all-reduce — a net steady-state win at long
  sequences. L01's whole six-update job was still slower on two (239.4 s vs
  248.0 s) because startup (model load, DDP setup, broadcasts) is paid twice;
  the win amortizes over longer jobs.

## D67 method (new measurement)

Gap filled: L01's 1.231× is a single point (one data mix, one length setting,
5 updates). The decision-relevant missing curve is **how the two-board LoRA
advantage scales with sequence length** — compute per update grows ~linearly
with length while the synchronized gradient payload is constant, so the ratio
should improve with length and cross 1.0 somewhere.

- **Fixture:** the qualified L01 integration (`train_two_host.py` +
  `bc250_lora_support.py`, identical hashes on both hosts, `832017cd…` /
  `2f24c6fe…`) in `BC250_LORA_BENCHMARK=1` mode: fresh output dirs, no
  evaluation/saving, every update asserted finite and not optimizer-skipped,
  per-update synchronized timing + global token counts + peak memory in
  `audit-rank*.json`.
- **Data:** owned synthetic pseudo-conversations (`make_owned_data.py`,
  deterministic seed 6701), **exactly** 1024 / 4096 / 8192 chat-template tokens
  per sample, 256 samples per length, byte-identical on both hosts (SHA-256
  verified). No user data read or copied. Exact lengths make per-update token
  counts identical across modes by construction.
- **Model/adapter:** real Qwen3.5-2B base (owned copy already staged on both
  boards), rank-128/alpha-256 all-linear LoRA, FP16 autocast + native scaler
  (init 128), eager attention, non-reentrant gradient checkpointing, dropout
  0.05, constant LR 1e-4, seed 42 — the L01 production-like settings.
- **Matched global batch 4:** single = micro 1 × accum 4; DDP = micro 1 ×
  accum 2 × 2 hosts. **1024:** 6 updates, first excluded (warm-up), remaining 5
  summarized by median/IQR; tokens/s = global tokens per update / median
  seconds. **4096:** cold first-update comparison only (1 update per mode):
  the first 4096 update alone measured 326.8 s single-board (eager attention
  O(L²) plus the reference gated-delta chunk loops), so no warmup-excluded
  steady-state measurement fits the qualified 460 s guardian child budget.
  **8192: not run — infeasible within the qualified safety envelope** (its
  first update would exceed the budget even before warmup exclusion; the
  owned 8192-token data was generated and hash-verified but no GPU case was
  attempted). L01's matched 8192-max-length measurement remains the
  long-sequence evidence. **Deviation from L01 (global 16):** the application
  environment's reference gated-delta path is launch-bound (the D66 root
  cause, unfixed in this newer Transformers environment): a global-16 update
  costs ~100–300 s at 1024, which cannot fit the 460 s budget with warmup plus
  measured updates. Global 4 keeps the 1024 cases inside the qualified
  envelope; D67 is internally matched, so the single-vs-DDP ratios are valid,
  but absolute times are not directly L01-comparable. The first 1024-single
  attempt at global 16 and the first 4096-single attempt at global 4 (4
  updates) were safely stopped by the guardian at the child deadline
  (acknowledged guards, clean boards) and are retained as evidence of the
  budget constraint.
- **Guardians:** adapted L01 guardian (`guard_d67.py`, surviving d09 preflight):
  SH runtime lock per host, 600 s boot-guard with nonce-matched ack, 460 s
  bounded child (4 CPUs, cpuset 0–3, 10 GiB RAM/12 GiB total, 512 PIDs, 256 MiB
  shm), kernel-log and distributed-registry identity, foreign-container refusal,
  owned-container cleanup. Root only for the guardian; containers run as the
  recorded user. Application originals, completed outputs and checkpoints
  untouched; all D67 paths additive under `bc250work/lora-bench-d67`.
- **Analysis:** `summarize_d67.py` (pure logic unit-tested); two-rank audits
  merged by max rank time per update with global-token agreement checks.

## D67 results

Owned synthetic data, exact lengths, global batch 4, rank-128 LoRA on the real
Qwen3.5-2B base, FP16. Full records: `kernel/artifacts/lora-bench-d67/`.

**1024 tokens — steady state (5 measured updates after warmup):**

| Mode | Median s/update | IQR% | tokens/s | Peak live GPU |
| --- | ---: | ---: | ---: | ---: |
| single (.104) | 36.262 | 0.22 | 113.0 | 6.16 GiB |
| two-host DDP (max rank) | 18.679 | 4.15 | 219.3 | 6.67 / 6.66 GiB |

**Throughput ratio: 1.941×** — near-ideal two-board scaling at this operating
point. DDP overlaps the 538 MB gradient all-reduce with the backward pass, so
the 1 GbE link is almost fully hidden. Per-update samples: single
36.29/36.23/36.30/36.19/36.26 s; DDP rank0 18.7/20.0/18.7/18.6/18.6 s, rank1
18.8/20.0/18.7/18.6/18.6 s (one 20 s outlier on both ranks inflates the IQR;
median robust). All updates finite, zero optimizer skips, both guardians
passed with kernel-log and registry identity on both hosts.

**4096 tokens — cold first update (directional, includes warmup):**

| Mode | Cold first update s | tokens/update | Peak live GPU |
| --- | ---: | ---: | ---: |
| single (.104) | 325.822 | 16384 | 6.97 GiB |
| two-host DDP (max rank) | 167.567 | 16384 | 7.47 / 7.48 GiB |

**Cold first-update ratio: 1.944×** — the same ~1.94× advantage as the 1024
steady state, i.e. the two-board benefit holds from the very first update at
this length (the 325.8 s single-board cold update reproduces the 326.8 s of the
stopped 4-update attempt). Both guardians passed with kernel-log/registry
identity on both hosts; all updates finite, zero skips.

**8192 tokens:** not run (infeasible within the qualified 460 s guardian child
budget; see method). L01's 1.231× at max-length 8192 with real mixed-length
data and global batch 16 remains the long-sequence reference point.

**Reading the scaling:** the two-board LoRA advantage is ~1.94× at both
measured operating points (1024 steady state 1.941×, 4096 cold 1.944×) — DDP's
backward/all-reduce overlap hides the fixed 538 MB gradient payload on 1 GbE
while the boards split the compute. L01's 1.231× at max-length 8192 with real
mixed-length data and global batch 16 is a different operating point (heavier
per-update compute mix, accumulation 4 per host), not a contradiction: the
ratio is workload-dependent. What does flip the result is **whole-job startup**
(model load, DDP setup, broadcasts — paid on both boards): L01's six-update job
was slower overall on two boards (248.0 s vs 239.4 s) despite the steady-state
win. Amortize startup over long jobs.

## Decision guidance

1. **Full-parameter training that fits one board:** use one board. Two boards
   are 6.8–68× slower per update at these sizes; choose FSDP only when you need
   the memory headroom and accept the throughput cost.
2. **Full-parameter training that does not fit one board:** two-board FSDP is
   the demonstrated path (1.024B FP32, D43). Expect ~53 s/update at seq 128 /
   global 2; plan long jobs and checkpoint at optimizer boundaries.
3. **LoRA fine-tuning of a large base:** two boards give ~1.9× per-update
   throughput at the measured operating points (D67), and L01 showed +23% on
   real long mixed-length data — but only for jobs long enough to amortize the
   doubled startup. For short jobs, one board is faster overall.
4. None of this combines the boards into one larger GPU; DDP does not increase
   per-board model capacity, and FSDP must still fit each gathered block plus
   activations and buffers per board.

## Boundaries

- Bounded owned-fixture measurements, not production, convergence or
  general-model speed guarantees; tiny/owned data, fixed lengths.
- 1 GbE Ethernet is the communication budget; no P2P/IB/SDMA; results do not
  transfer to faster interconnects.
- The frozen 5.2.0 stack fixtures and the 5.22.1 application environment are
  distinct; D67 measures the application environment only.
- Reference causal-conv/gated-delta paths remain in the application environment
  (the D66 native route is a frozen-5.2.0 stack path, not applied here).
- No runtime/default/package/source change; no kernel/boot change; both boards
  idle before/after; .105 production healthchecks and unrelated apps preserved.

## Verification and evidence

- 24 new workstation tests (`bc250_lora_bench_d67_test.py`); full bc250 suite
  **249 passed**; Black clean.
- Generator defects found and fixed while qualifying: shared-RNG non-monotonic
  search; fixed prompt fraction overshooting the token budget for multi-token
  syllables; transformers 5.22.1 `apply_chat_template` returning a
  `BatchEncoding` (UserDict, `len` == key count) — normalized and regression-
  tested, including the unreachable-total fallback (prompt word count ×
  completion text variants).
- The first (global-16) 1024-single attempt hit the 460 s guardian child
  deadline and was safely stopped (guard acknowledged, owned container removed,
  registry unchanged, no foreign containers). Its kernel-log identity gate
  flagged one host-level line: `Adding 4194300k swap on /swapfile` — the
  pre-existing (2026-10-22) swap file was activated during the run by an
  unidentified host actor; no GPU/kernel fault lines, board healthy afterwards
  (13.2 GiB available, 22 MiB swap used). The attempt's records are retained in
  `kernel/artifacts/lora-bench-d67/attempt-global16-1024-single/`. The first
  global-4 4096-single attempt (4 updates) likewise stopped safely at the
  deadline after its 326.8 s first update
  (`attempt-4096-single-global4/`); all subsequent cases ran against the
  post-activation kernel-log baseline and passed the identity gates.
- Owned data: 256 samples per length, exact chat-template token counts,
  byte-identical on both hosts (generator `d706e066…`; per-length SHA-256 in
  `data-manifest-rank{0,1}.json`, manifests equal). The 8192 data was generated
  and verified but no 8192 GPU case was attempted (budget infeasibility).
- Evidence: `kernel/artifacts/lora-bench-d67/` — per-case guardian
  final/running records, rank audits, `summary.json`, data manifests, stopped-
  attempt records. Host task directories `bc250work/lora-bench-d67` retained on
  both boards for replay. Both boards idle at handoff (0 containers, 0 GPU
  clients); no armed guards remain; application originals/completed
  outputs/checkpoints untouched; no user data read or copied.
