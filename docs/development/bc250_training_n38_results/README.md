# N3.8: vocabulary-projection audit

2026-10-19. Implementation `d7b7a2bc4`; explicit settling `ed0946ef6`;
resource preflight `f85831f3b`. This audits actual Qwen3.5-2B vocabulary forward
and input-backward GEMMs on native gfx1013. The experimental CSV is retained
for reproducibility; the installed N3.2 preset remains unchanged.

## Entry identity and application drift

The original N3.7 baseline refused before a GPU run: two application files had
changed since qualification. Full inspection of all311 protected paths found:

- `replica/train.py` SHA256 `170d39204518bcd9526f86dba3f67c40cfeafbd547f764796e366dbb9e02ac8e`
  removes N3.7 provenance integration and the A07 attention-backend CLI/tiled path.
- `replica/bc250_ops.py` SHA256 `d058b14a4bae969e5b84a9de8f69706c0f6e3df910409dfc15843f0eeee3d41a`
  changes only its printed status; its arithmetic matches the qualified hook.

The other309 paths match. Both current files are preserved, archived locally,
and explicitly admitted in `WS/logs/training-n38-entry-r2/final.json`.
That record is an entry snapshot, **not application qualification**. Current
production trainer therefore must not be described as enforcing N3.7 provenance
or exposing the A07 tiled CLI. Those implementations remain tracked locally.
N3.8 isolated model workers load a separate byte-exact qualified hook
`n38_qualified_bc250_ops.py`, SHA256
`f0bcb35eadd3cde6c08dbeca0929358e4361f1f2b7358f2cad5288517f9fa4e6`.
No user application/checkpoint is rewritten.

## Layouts, independent references and policy correction

One real synthetic1024-token/batch1/rank16/alpha32/BF16 forward/backward captures:

| Role | Torch A @ B | A/B strides | rocBLAS layout |
| --- | --- | --- | --- |
| Vocabulary forward | [1024,2048] @ [2048,248320] | [2048,1] / [1,2048] | T,N; M248320,N1024,K2048 |
| Input backward | [1024,248320] @ [248320,2048] | [248320,1] / [2048,1] | N,N; M2048,N1024,K248320 |

Both inputs are BF16, offsets0; base frozen, all372 adapters nonzero/finite.
FP32 adapters/math SDPA/non-reentrant checkpointing and N3.2 override retained.
The two owned fixtures total2,546,995,098 bytes and are SHA256-verified both
remotely and in durable local `kernel/artifacts/training-n38/fixtures`.

Full independent CPU FP64 references use256 vocabulary columns per tile for
forward, and sums of2048-wide K products for backward. Owned FP64 tensor bounds
are2,057,306,112B and67,108,864B respectively; these are tensor bounds, not process
RSS. Output comparison uses small tiles, unchanged atol0.05/rtol0.015 and global
relativeL2<0.004.128-element guards and immutable full inputs are checked.
Five small forward-gate CPU contracts cover uneven reference tiles and rejection
of NaN/incorrect near-zero output. All14 admitted scalar solutions plus the
default pass for each role, both in initial and matched-policy audits.

The first private rocBLAS handle used atomics mode0. Actual eager Torch uses
atomics allowed and32MiB workspace here, confirmed by the captured bench log
and qualified local `CublasHandlePool.cpp` (workspace/atomics setup); native
`handle.hpp` initializes mode0. Consequently the original private backward
2.443x ratio is **not a Torch speedup**. Corrected runs explicitly match Torch's
policy and include paired actual `torch.mm` timings. Actual Torch output equals
the matched private default exactly; selected candidates differ in reduction
order and are not bitwise replacements.

| Matched-policy role | Candidate | Torch median ms | Candidate median ms | Isolated ratio |
| --- | --- | --- | --- | --- |
| Forward | -339 | 370.895 | 343.939 | 1.078x |
| Input backward | -533 | 373.431 | 216.697 | 1.723x |

Six alternating rounds, three calls per timing; native warm calls synchronize.
Worst scalar relativeL2 is approximately0.00167. Selection is limited to these
actual layouts/dtypes; no universal GEMM, sustained-throughput or LoRA claim.

## Full-model gate and resource failure

Combined experimental CSV keeps N3.2's -490 row and adds only the two vocabulary
rows above. Four fresh ABBA workers compare first-step loss/all372 clipped
adapter gradients against the original reference. Unchanged contracts are
abs loss<0.015 and max gradient relativeL2<0.04; each worker independently checks
its own parameter update and AdamW step/exp_avg/exp_avg_sq against CPU AdamW.
CPU/GPU RNG hashes must match, and the final reference repeats first loss and
all372 gradients exactly. This is not a CPU reference for the entire model.

Original r7 passes these contracts: worst adapter relativeL2=0.0144596;
max CPU parameter error1.86265e-9. Timings shift strongly between early/late
workers (reference medians7.806/5.649s, candidates6.593/5.354s), so this result
alone does not establish a repeatable whole-model speedup.

The r8 settling attempt fails: global host OOM kills its first Python worker
(SIGKILL9); no completed model gate. A separate UID1000 `train` process appears
in the OOM task list. Concurrent activity is recorded without claiming exclusive
causality. Kernel logs differ and the failed evidence remains failed. This is
neither the unresolved Python CPU segfault nor the previously fixed VM-heap
corruption. Its own container is cleaned and600s guard acknowledged. No other
process is stopped. After the user reserves the board, r9 requires >8GiB host
MemAvailable and no other GPU holders except the existing root governor.
Fixture export finishes before timing; two extra untimed full steps precede
three measured steps in every fresh worker.


r9 passes all four ABBA workers and the strict kernel-log/protected-identity
controller. Reference medians are5.8094/5.4350s; candidate medians5.3520/5.3524s.
Paired ratios are1.08547x and1.01542x. First loss is identical14.2942647934;
worst adapter gradient relativeL2=0.0144595988, CPU AdamW max parameter error
1.86264515e-9, RNG matches; final reference first gradients/loss repeat exactly.
The remaining reference timing shift prevents a stable whole-model gain claim.
No measured loss/gradient condition was weakened to admit the candidates.

## Decision, preservation and next action

**No candidate deployment.** Keep the installed N3.2 CSV SHA256
`1f2947a88ee69a3d3854ec7615f209dadd19292489b7df3d101c5941727846c0`.
Experimental CSV SHA256
`ffc1b5a3f61dc8fddf2d25627558dcb5fb3621742af56c23a0ab0bf21d81f3e9`
is available at `build_tools/bc250/rocblas_gemm_n38.csv` but is not a recommended
new default. Narrow isolated timings do not establish actual accumulated LoRA
throughput, convergence or bitwise checkpoint continuity under changed reduction
order. Square-projection candidate stays rejected/inactive too.

14 composite checks pass across layout2/raw audits2/matched audits2/model4/
settled-model4. Completed model gates execute41 synthetic training micros
(layout1 + original ABBA16 + settled ABBA24); failed r8 work is excluded.
All eight600s guards acknowledge; successful gates have identical dmesg, while
whole-batch dmesg differs because of r8 host OOM. Preserve both failures:
r1 preflight (no GPU) and r8 resource failure. r9 passed after explicit user
reservation; resource preflight records15,211,716,608B MemAvailable and only the
root board governor as GPU holder.

All311 protected files remain at the explicitly recorded entry identities;
the four FFT AOT databases, active links and selected native libraries remain.
Runtime62ab2437c/rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4/other libraries
319a235d6 retained. Same7.1.5-bc250rocm-r2 bootd3857b96; no native build/install,
cache reset, removal, reboot, BIOS/fallback/.105 change. Native build resource
settings/ccache are not invoked because only qualification helpers changed.
Shared model helper is backed up at
`WS/recovery/training-n38-before/{validate_model_gemm_n32.py,validate_model_gemm_n32.pre-settle.py}`;
its new keyword hook path and optional settling default preserve old callers.
Black on six helpers, Ruff-F on seven sources, syntax and diff checks pass.
Byte-exact qualified copied hook is deliberately not reformatted.

Final disk available9,799,057,408B observed. Own jobs/containers/guard deadlines
are finished; no other workload was killed or edited. Unknown Python CPU fault
remains unresolved and separate from repaired VM-heap/CSR and this host OOM.

**Next N3.9:** compare preserved current trainer against qualified N3.7/A07
sources, reconcile provenance and attention CLI integration without dropping
current user edits. Use the unchanged N3.2 preset and owned masked1024/rank16/
accum16 fresh/resume/refusal gates before installing any reconciled trainer.
Do not silently rewrite/migrate existing new-provenance checkpoints or activate
the experimental vocabulary CSV as part of that repair.

## Reproduction and artifacts

Rootful commands on .104 are run through the existing exclusive lock:

```
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n38_on_host.py --stage model-stable --attempt r9
```

Replace `WS` with `/home/nik/bc250work/rocm-10.2-gfx1013`.
Other stages: r2 layout, r3 forward, r4 input-backward, r5 policy-forward,
r6 policy-backward, r7 model; r8 model-stable is the preserved OOM failure.
The entry recorder ran under that same lock. r1 used the prior N3.7 baseline
and refused changed application source before container start.

Raw logs, JSON, kernel logs, commands, guard states and eight passed-worker372-gradient
archives plus the failed r8 worker gradient archive remain at `WS/logs/training-n38-*` and durable local
`kernel/artifacts/training-n38/remote`. Fixtures are on disk (not tmpfs) and
verified in local `fixtures/`. Exact first raw benchmark source and pre-settle
r7 sources remain locally in `source-initial/` and `source-r7/`; committed code
history retains subsequent executed variants. Original application entry files
are in `application-entry/`. All136 raw regular files (3,159,740,816B) are SHA256-verified locally;
compact raw evidence and verification are beside this report. No user dataset/checkpoint was used or overwritten.
