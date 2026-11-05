# D19: compact metric validation and vocabulary crossover

Final selected helper/fixture revision `1a39b5254`; original dense prototype
`a7c1104d0`, wide-reference fixture revision `4deb6a00a`, export/verifier
`c99e46fe4`. [Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#compact-metric-validation-and-larger-vocabulary-d19).
[Verified final observations](verified-results.json).
[Rejected dense prototype observations](dense-prototype-verified-results.json).

The opt-in source helper is staged on both hosts at
`/home/nik/bc250work/tools/distributed-d19-r3/bc250_compact_token_metrics.py`.
It retains D18's per-row active-token selection and bounded FP64 CE temporaries,
but accumulates validation on the device and reads one final scalar flag.
Active invalid labels are internally clamped before device indexing, then
refused before any result is returned. Ignored nonfinite logits are omitted;
active nonfinite logits are refused. CPU/GPU tests cover both aligned masked
and once-shifted causal targets, zero-target rows, chunk sizes1/7/4096, label
range and nonfinite refusals. Dataset callbacks remain stateless.

CPU dispatch profiling of the same3-row GPU probe observes scalar reads
12→1, while dynamic `aten::nonzero` calls remain3. This is not elimination of
all synchronization or physical kernel/counter profiling. No new backward,
training throughput, convergence or arbitrary model qualification is claimed.

## Final qualification

Final r3 repeats the six small Bert/GPT2 FP32/BF16/FP16 cases and adds
Bert/GPT2 FP32 with32768 vocabulary and maximum sequence64. Models are owned,
randomly initialized, eager/dropout0/cacheFalse; no downloads or application
access. Each mode is warmed before matched full/D18/D19/D19/D18/full prediction.
Native Trainer.evaluate, labeled predict, and explicit full-logit label-free
opt-out are exercised. Model parameters are on GPU; Linear/Conv1D output dtypes
are checked. There is no optimizer; all model state values remain unchanged
and gradients remain None. Native evaluation mode is intentional, not full RNG
rollback. Native loss is EXACTLY unchanged between all modes.

Nineteen variable-length right-padded examples, unequal prompt/target masks,
two completely ignored first rows, and a final repeated distributed sample test
native gathering/de-duplication. Exact19 returned labels in global order;
small masked126/causal120 and wide masked838/causal832 active targets. Per-device
evalbatch2/workers0/accumulation-to-CPU2/concatenation retained, world2/RCCL.
Compact arrays are[19,3] FP64 sums/counts/correct; full arrays are[N,L,V] FP32.
Metrics match full-logit and manual NumPy64 logsumexp/argmax references within
atol/rtol1e-11. Every real-position full vocabulary logit also passes the CPU
FP64 model comparison: FP32 atol3e-6/rtol5e-5, BF16 .003/.003, FP16 .001/.001.
The CPU model is the same pinned HF graph with different arithmetic, not an
independently implemented Transformer. All ranks agree. No tolerances relaxed.

Final176 check groups/16 accepted recovery guards, five CPU contract patterns
observed80 times and two GPU contract patterns observed32 times. These are
final-r3 counts only, not combined with previous prototype runs. Black,
critical Ruff+F401, syntax/diff checks and42 local regressions pass; no full
upstream CI. Source/native identity/config hashes, per-phase guards/ACKs,
unchanged kernel logs and registries, clean successful agent exits, cross-rank
metrics/full prediction digests and archives are verified. Archive validation
does not reexecute GPU math. Native all_gather/barrier and profiler messages
remain visible, not suppressed.

## Rejected dense prototype and unresolved BF16 gate

The initial rectangular batching prototype removed nonzero synchronizations but
processed ignored/padded token rows. Wider FP32 latency regressed to78ms versus
59–61ms for frozen D18, and peak allocator rose to100.7–101.0MB versus60.9–61.2MB.
It is preserved under tools/evidence distributed-d19 r1/r2 and was NOT selected.
The r3 active-token version retains the needed validation reduction without that
padding work. Reduced scalar reads alone are not evidence of a latency win.

Initial Bert32768-vocabulary BF16 full/D18/dense-prototype metric parity passed,
but the unchanged D17 all-logit CPU FP64 gate FAILED:2 of30,408,704 compared
values, maximum absolute difference0.003654425659018067, fixed atol/rtol.003/.003.
Both hosts preserve the raw exception and acknowledged recovery guards; no
kernel-log/registry change accompanied it. GPT2 wide BF16 was not run after the
coordinator stopped. This is an unresolved wider-BF16 numerical qualification,
not a demonstrated native GPU defect or an excuse to relax the gate. Wider FP32
was selected for the strict-reference crossover; it does not fix or qualify
those BF16 outliers. Small BF16 scope remains passing. These two failed guards
and16 successful earlier prototype guards are retained separately from final16.

## Measurement limits

Tables below use two measurements per mode, after warming ALL three modes.
They are bounded paired observations, not a statistical benchmark. CPU barriers,
CUDA synchronization, wall time and reset live Torch allocator peaks are recorded.
Prediction payload counts are INPUT TENSOR BYTES summed over five native gather
calls per rank, not NIC wire bytes or all distributed communication. Native
padding collectives, gathered labels/losses and protocol overhead remain.
Dataset prediction arrays are CPU objects; their size is not a total GPU-memory
saving. The fixture retains full CPU reference arrays and computes a costly
list-based prediction fingerprint OUTSIDE timed prediction, so no host RSS or
whole-fixture time/memory gain is inferred. No persistent prediction tensors or
checkpoints are created. No large pretrained model fit or training speedup proof.

The helper still materializes local batch logits and native sequence padding;
labels/losses are gathered normally. Chunk tokens caps active positions per
FP64 temporary, not bytes independently of vocabulary or total process memory.
Both metric callbacks must be opted in together; disable both for full logits
and label-free prediction. No streaming/batch_eval_metrics/generation/arbitrary
output tuple/IterableDataset/multiworker/eval+train resume/elastic qualification.
Rank-asymmetric refusal is not collective fault recovery; guards remain required.

## Reproduction and handoff

Owned final coordinator on BOTH idle hosts, as root:

```sh
python3 /home/nik/bc250work/tools/distributed-d19-r3/run_distributed_d19_on_host.py
```

It refuses existing evidence. Preserve current results rather than rerunning over
them. Exact commands/PIDs/exits live in `evidence/distributed-d19-r3/sequence.json`;
phase directories contain rank.json/final.json/gpu.log/kernel logs/guards.
Controller log: `evidence/distributed-d19-r3-controller.log`.
Controllers8981(.104)/8659(.105). Earlier controllers4885/4969 stopped with exit1
on BF16 gate; r2 controllers7617/7411 completed the two FP32 wide cases. Shell
wait for .104 r2 returned127 after its completed job had been reaped; its phase
journals and all final guard records independently show success.

Resource/recovery settings unchanged: SH host runtime locks, one gfx1013 GPU per
node,4 CPUs/cpuset0–3/10GiB RAM/12GiB memory+swap/512 PIDs/256MiB shm,600s recovery
guards/460s child deadlines. Native daily/RCCL Torch a524980/ROCm/D09 launcher/
consumer packages/profiles/images/registries/kernels/defaults are unchanged.
No native build/install/prefix mutation/reboot/model download/user workload stop.
Qualified boots d83af4e1-e3ef-4cc8-ade9-371328a22f4b(.104r2),
bad537ed-ea08-49e6-ad44-40a9bc1d6914(.105r3). Preserve SSH and boot fallbacks.
Historical Python CPU fault/KFD/invalid visibility remain separate from fixed
VM heap corruption and fixed D05 agent exit. No jobs/containers/armed guards
remain at handoff. Both host docs and memory snapshots are synchronized.

**Next D20 (planned):** diagnose wider BF16 Bert logit outliers using matched
CPU-autocast and FP64 references at the original gates. Distinguish normal BF16
arithmetic from a reproduced port defect before changing code or qualification.
Also replace the owned fixture's expensive list fingerprint with a byte digest
before another large-vocabulary run; this does not change model math.

## Final measurements

Small fixtures: paired D18/D19 latency ratios1.014–1.034 (two samples per mode),
not statistically established gains. Wider fixtures are essentially level with
D18 (D18/D19 ratios0.990–0.999), so no wider speed gain over D18 is claimed.
Compact versus FULL-logit evaluation is18.6–19.6x faster in these two wider FP32
fixtures, about57.7–61.3ms versus1.12–1.15s. That gain is from compact evaluation
already available through D18, now measured at this vocabulary, not a new whole-
Torch or LoRA training multiplier. Some small full-mode runs have timing outliers;
full mode is normally faster than compact in these small-vocabulary examples.

Wide returned predictions159,383,552 bytes(152MiB)→456 bytes, labels9,728 bytes
remain; prediction-gather input79,167,488→240 bytes per rank over five calls.
Wide live allocator peaks drop roughly75.5MB, not152MiB: model batch logits and
native padding remain. Final D19 peak is within1,024–1,536 bytes of D18 in these
wide cases, rather than the rejected prototype's ~40MB penalty. Tensor payload
ratios do not describe total Ethernet traffic or model capacity.

| Case | Host | Mean full / D18 / D19 ms | Peak live allocator full / D18 / D19 bytes |
| --- | --- | --- | --- |
| bert-fp32-eval | .104 | 38.78 / 45.84 / 44.32 | 307712 / 267264 / 267264 |
| bert-fp32-eval | .105 | 38.73 / 45.88 / 44.35 | 307200 / 270848 / 270848 |
| bert-bf16-eval | .104 | 65.84 / 49.18 / 48.07 | 312832 / 300544 / 300544 |
| bert-bf16-eval | .105 | 42.56 / 49.20 / 48.07 | 317440 / 305152 / 305152 |
| bert-fp16-eval | .104 | 42.58 / 49.60 / 48.68 | 312832 / 300544 / 300544 |
| bert-fp16-eval | .105 | 42.57 / 49.51 / 48.63 | 317440 / 305152 / 305152 |
| gpt2-fp32-eval | .104 | 37.91 / 44.03 / 43.23 | 297984 / 268800 / 268800 |
| gpt2-fp32-eval | .105 | 37.83 / 44.02 / 43.17 | 297472 / 276992 / 276992 |
| gpt2-bf16-eval | .104 | 63.87 / 47.82 / 46.37 | 305664 / 293376 / 293376 |
| gpt2-bf16-eval | .105 | 63.22 / 47.73 / 46.37 | 311808 / 299520 / 299520 |
| gpt2-fp16-eval | .104 | 63.33 / 46.57 / 45.93 | 305664 / 293376 / 293376 |
| gpt2-fp16-eval | .105 | 41.34 / 46.52 / 45.84 | 311808 / 299520 / 299520 |
| bertwide-fp32-eval | .104 | 1119.21 / 57.22 / 57.71 | 136687104 / 61188608 / 61190144 |
| bertwide-fp32-eval | .105 | 1132.66 / 57.23 / 57.79 | 136688128 / 61090304 / 61091328 |
| gpt2wide-fp32-eval | .104 | 1140.51 / 61.21 / 61.25 | 136545792 / 61014528 / 61016064 |
| gpt2wide-fp32-eval | .105 | 1147.24 / 61.25 / 61.31 | 136546816 / 60850688 / 60851712 |

Final workstation evidence archives under `kernel/artifacts/distributed-d19`:

- .104 r3:1,013,179 bytes, SHA256 `bbce25bca5ccba8c8474f0e6e0ee183480da2f463c619c10a92b5c50415b1d39`.
- .105 r3:914,431 bytes, SHA256 `e565fcbb83b7a58b1ff4c26033ae6f6129481bac737f7b4251e58fb2607fc8ea`.
- Earlier prototype archives remain `104-evidence-r1.tar.gz` (553356 bytes,
  SHA256 `68b847a6e1663c85661d66e0e3864017897dc31543ba6dc2f2f75c87f31a1f34`)
  and `105-evidence-r1.tar.gz` (501383 bytes,
  SHA256 `96ee0744fc9154a9bbb957f663d371a28c116dfc36f6f740bdc5ee294c89ecd7`).

Final r3 archives also retain all r1/r2 source/evidence trees and failed BF16
logs; previously verified prototype summary is preserved separately. Both final
coordinators exit0. Host docs and memory copies match local sources byte-for-byte.
