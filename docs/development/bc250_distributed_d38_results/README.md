# D38: native CPU checkpointing with exact gradient streaming

Implementation `22d702e47`, additive `~/bc250work/tools/distributed-d38` on BOTH
hosts. This extends the CPU oracle, not ordinary GPU training. The fixed352.96M
FP64 GPT2/full-global2/sequence128/positions256/dropout0/cacheFalse recipe and
D35 raw reference files are retained. CUDA stays uninitialized; no GPU tensors,
optimizer steps or training process group run in these workers.

## Native recipe and proof

The checkpoint path calls native `model.gradient_checkpointing_enable` with
`use_reentrant=False,preserve_rng_state=True`; the paired streaming control keeps
checkpointing off. All13 checkpoint-capable modules must have the intended live
flag. Enabled functions must be argument-free partials of the actual native
`torch.utils.checkpoint.checkpoint` with the exact boolean kwargs. Missing kwargs,
truthy integer substitutions, changed RNG/reentrant flags, wrong functions and
unexpected positional arguments refuse. Recipe/function guards run after enable,
before forward and after backward.

Forward-pre hooks count actual block entries: each of12 blocks enters once in
forward; checkpointing enters each again during backward, versus no additional
entries in the control. Non-reentrant early-stop can finish recomputation before
the complete forward returns; counts prove entry/recomputation, not two complete
forward executions. All148 post-accumulate leaf hooks still complete exactly once
and release `.grad`; zero parameter gradients remain after backward. Original
weights are checked directly in FP64 before/after, CPU loss and ALL gradient
bytes/SHA256 match D35/D37 and both modes/hosts EXACTLY. CPU loss remains
9.364227904214559. CPU RNG snapshots are unchanged in this dropout-zero fixture;
no new stochastic/dropout checkpointing claim follows from that observation.

The installed checkpoint source matches qualified `torch/torch/utils/checkpoint.py`
byte-for-byte, SHA256
`eab49639231cc989d65c04212923eef0ade4aa059b68bcabfe2819007f34e69f`.
The offline verifier pins this immutable identity. Native Torch/D09 package/agent
identity and the three Transformers modeling pins remain checked. CPU oracle
sources are bound separately from the new checkpoint/streaming implementation.

## Results (2026-11-09)

[Verified results](verified-results.json):32 executed CPU groups/four ACKed guards,
86 local tests,Black/critical Ruff+F401/diff pass. Three new local policy tests and
three private source/recompute-count/recipe metadata corruptions refuse. The two
metadata cases synchronize rank/final/handoff copies so the semantic guards are
exercised. Original evidence remains unchanged. No failed CPU phase, relaxed
numerical bound or full upstream CI claim. D35 and D37 published archives are
hash-verified and replayed as provenance; D38 controls match D37 exact values.

| Host | Cgroup peak, streaming→checkpoint+stream | Peak process RSS | Host available minimum |
| --- | --- | --- | --- |
| .104 | 5208.641→4179.109MiB | 5149.285→4127.707MiB | 9.237→10.261GiB |
| .105 | 5196.531→4249.711MiB | 5139.398→4200.547MiB | 9.132→10.033GiB |

Checkpointing lowers measured CPU cgroup peak by18.2–19.8% beyond streaming alone
in this paired fixture. Scratch remains at most96MiB, one owned raw file, with
5GiB disk reserve. No new full reference set or full gradient spool. After-backward
cgroup usage3615–3697MiB; full FP64 parameters/runtime buffers remain resident.
Counters overlap and must not be summed. Kernel cgroup peak/VmHWM supplement200ms
samples. Host/cgroup swap and OOM counters stay0; kernel logs/registry unchanged.

Complete guarded phases are71s control/80s checkpoint; worker durations57–58s/
58–66s include source/reference validation, recomputation, checks and I/O. One
paired order is enough for this exactness/recompute scope, not a timing-significance
or training-speed claim. This is CPU-oracle memory reduction, not a native GPU
training-memory improvement, 1.024B fit proof, arbitrary-model or long-run guarantee.

## Replay

Preserve completed series `~/bc250work/evidence/distributed-d38-r1`. On a clean
replay target with the original D35 references, coordinate BOTH idle boards:

```bash
doas python3 ~/bc250work/tools/distributed-d38/guard_distributed_d38_on_host.py \
  --checkpoint off --label stream-control --port 31610
# After successful controls on BOTH:
doas python3 ~/bc250work/tools/distributed-d38/guard_distributed_d38_on_host.py \
  --checkpoint nonreentrant --label checkpoint-stream --port 31611
```

The installed D09 launcher coordinates two independent CPU workers. PLAN-01 EX
cooperative reservation+SH runtime lock, idle workload preflight,4CPUs/cpuset0–3/
10GiB RAM/12GiB combined RAM+swap/512PIDs/256MiBshm/600s recovery/460s child remain.
CPU admission≥12GiB host available,96MiB scratch+5GiB disk; ACK follows owned exit.
No unrelated process stop. Original D35 raw files stay read-only and are rehashed
at export; archives contain metadata/source/logs, not tensor payloads.

```bash
python3 therock/build_tools/bc250/verify_distributed_d38_exports.py \
  --host104 kernel/artifacts/distributed-d38/104-evidence-r1.tar.gz \
  --host105 kernel/artifacts/distributed-d38/105-evidence-r1.tar.gz \
  --baseline104 kernel/artifacts/distributed-d35/104-evidence-r1.tar.gz \
  --baseline105 kernel/artifacts/distributed-d35/105-evidence-r1.tar.gz \
  --baseline-results therock/docs/development/bc250_distributed_d35_results/verified-results.json \
  --stream-baseline104 kernel/artifacts/distributed-d37/104-evidence-r2.tar.gz \
  --stream-baseline105 kernel/artifacts/distributed-d37/105-evidence-r2.tar.gz \
  --stream-baseline-results therock/docs/development/bc250_distributed_d37_results/verified-results.json \
  --output therock/docs/development/bc250_distributed_d38_results/verified-results.json
```

Guardians104:5110/5946,105:19068/20336; exports6842/21712 all exit0. Exact commands,
CIDs/UUIDs/start ticks/ACKs/logs remain in each phase. Local artifacts:
`kernel/artifacts/distributed-d38/`. Final idle preflight passes BOTH, no owned
jobs/containers/armed guards. Boots2d86d523(.104/r2),e902e636(.105/r3) unchanged.
Native PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/image/defaults/fallbacks retained.
No native build/reboot/download/prefix mutation/application inspection. Historical
Python CPU fault stays distinct from fixed VM heap/D05 exit/D21 optimizer restore.

Next D39: qualify independent CPU-reference clipped-gradient checks in the D36
per-unit GPU audit, replacing the full actual-GPU-gradient spool at the unchanged
1e-8 absolute/5e-5 relative clipping bounds. Compare all raw/clipped/updated/local
AdamW digests with the qualified legacy route on352.96M, preserving the fallback
and original references. If a gate fails, investigate rather than relax it.
The unchanged1.024B actual-gradient spool would need3.816GiB plus5GiB disk reserve,
which .105 lacks. Removing it is one storage prerequisite; larger oracle/reference
storage and CPU/GPU memory admission remain separate.1.024B is still unqualified;
PLAN-05 remains OPEN, six important packages remain.
