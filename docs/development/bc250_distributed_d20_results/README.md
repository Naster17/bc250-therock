# D20: BF16 outlier diagnosis and explicit FP32 evaluation head

Completed 2026-11-05. Helpers/contracts/fixtures `1a61687a7`, export/verifier
`997b38bb7`, final scope/export `570e60428`. [Verified diagnostics](verified-results.json).
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#explicit-fp32-evaluation-head-d20).

D19's two wide Bert BF16 logit outliers are reproduced EXACTLY by matched CPU
BF16 autocast at the same coordinates and values. GPU versus matched CPU BF16
has zero violations at the unchanged atol/rtol.003/.003 across30,408,704 real-
position vocabulary logits. This supports BF16 rounding relative to FP64 as the
cause in this fixture; no native ROCm/Torch arithmetic defect was demonstrated.
It is not universal correctness proof or a reason to change the old tolerance.
Native BF16 still fails that strict FP64 gate on this fixture; its record remains.

Added `bc250_eval_precision.fp32_eval_head`: an opt-in evaluation context for ONE
existing FP32-master head with a single positional hidden-state tensor input and
one tensor output. It disables autocast within that head and promotes the input
to FP32, without changing parameters, buffers, tensor values, keys or ties. Caller
selects the head explicitly, requires all head submodules in eval mode, and runs
with no_grad/inference_mode. Per-instance forward is restored on scope exit,
including exceptions. Nested scopes/training/low-precision weights/invalid input
are refused. No global Torch, Transformers, AMP or application patch is installed.

Actual BertMLM `model.cls` head in the owned fixture now passes the ORIGINAL
FP64 gate with zero violations. Encoder AMP scope remains enabled; the precision
context encloses only the head. It intentionally changes head arithmetic/output
values relative to native BF16. It is not bitwise continuation or a training/
autograd policy; do not apply this helper to a permanently BF16-weight model.

| Comparison | Native BF16 | BF16 AMP + explicit FP32 MLM head |
| --- | --- | --- |
| GPU versus CPU FP64, violations | 2 | 0 |
| Maximum absolute logit error versus CPU FP64 | .003654425659018067 | .00005873836473679894 |
| Relative L2 versus CPU FP64 | .00423715627 | .00006318261 |
| GPU versus matched CPU autocast, violations | 0 | 0 |
| Maximum absolute error versus matched CPU | .001953125 | .000004708766937255859 |
| CPU autocast versus FP64, violations | 2 | 0 |
| token_nll | 10.432745588959701 | 10.43274511247336 |

Native outliers: record13/position42/vocab16960 gives GPU=CPU-BF16
.007293701171875 versus FP64 .010948126830893067; record9/position38/vocab30852
gives GPU=CPU-BF16 .140625 versus FP64 .14417412714823571. Matched microbatches
recreate both native ranks' local sizes/padding and duplicated tail, not a single
CPU BF16 whole-dataset batch. Layer taps at the first outlier show matching
BF16 dense/GELU outputs, differing native CPU/GPU LayerNorm outputs and close
final logits; they do not independently prove the entire model graph. Final
CPU reference uses the same pinned HF graph with different CPU kernels.

## Tests and boundaries

Three guarded phases on each host: native diagnosis r1, explicit-head r2,
actual evaluate/scope restoration r3. All use
BertForMaskedLM/vocab32768/hidden32/2layers/4heads/intermediate64/positions128,
eager attention/dropout0/cacheFalse/FP32 master weights/BF16 AMP. Original seed773,
19 owned records, actual lengths33–64/right padding/unequal target masks and two
ignored first records. Native Trainer predict/eval world2/RCCL, per-device2,
workers0/eval_accumulation_steps2/concatenation; exact19 label rows/global order
and838 active targets. Native source-pinned Trainer5.2/Accelerate1.12 helpers
and selected native provider retained. No new Transformer/backward/optimizer/
FSDP/compilation/generation/capacity qualification.

36 GPU check groups /6 acknowledged recovery guards; three digest CPU contract
patterns observed12 times, four head-scope CPU patterns observed8 times. Local42
regressions, Black, critical Ruff+F401, syntax/diff checks pass, no full upstream CI.
Full/compact token metrics agree within1e-11 and native loss is unchanged WITHIN
each precision policy. Changing head precision intentionally changes model math
and can change loss. All state values unchanged/gradients None, no optimizer.
The GPU tests observe FP32 head output, BF16 encoder query output, exact native
BF16 outlier restoration, and identical native Trainer.evaluate metrics after
normal and exception scope exits. Model parameter identities/state keys and
embedding/decoder weight ties are preserved. Both ranks have identical versioned prediction byte digests.
Native evaluation mode is intentional; no whole-process RNG rollback claim.

Added `bc250_array_digest.array_digest` and updated the owned D19 fixture to stop
materializing full prediction Python lists/JSON for fingerprinting. Versioned
SHA256 includes names/shapes/dtypes and exact logical C-order numeric bytes.
Chunked NumPy iteration handles strided inputs without a whole contiguous copy;
CPU tests compare with an independent whole-byte oracle at chunk1/3/65536,
ordering/layout equivalence, dtype/shape/signed-zero sensitivity and object/
structured-array refusals. This is a NEW digest format; legacy JSON digests are
not comparable strings. Default chunk65536 bounds a copied F32/I64 buffer to
256/512KiB; references/full logits still exist. No paired speed or host-RSS
measurement claim. No persistent prediction/checkpoint tensors were created.

Helpers staged BOTH under `tools/distributed-d20-r3`; earlier r1 tools/evidence
preserved. Daily/RCCL Torch a524980, ROCm, D09 launcher, frozen consumers, native
profiles/images/registries/kernels/defaults unchanged. No native build/install/
prefix mutation/reboot/model download/application access or foreign workload stop.
Historical Python CPU fault/KFD/visibility remain separate from fixed VM heap
corruption and fixed D05 agent exit. Native all_gather/barrier warnings retained.

## Reproduction and handoff

As root on BOTH idle hosts, final helper/fixture path:

```sh
python3 /home/nik/bc250work/tools/distributed-d20-r3/guard_distributed_d20_on_host.py \
  --case bertwide-bf16 --action fp32-head --series r2 \
  --label bertwide-bf16-fp32-head --port 30102
```

Do not rerun over preserved evidence; it refuses existing phase directories.
Initial diagnosis command used tools/distributed-d20, action diagnose, seriesr1,
labelbertwide-bf16-diagnose, port30101. The r2 full-reference head test used
tools/distributed-d20-r2 and action fp32-head/port30102; r3 added action scope/
seriesr3/labelbertwide-bf16-scope/port30103 using the same final helper.
Initial archives remain preserved as104/105-evidence-r1.tar.gz. Root guards record exact worker commands,
container/guardian IDs and runtime/launcher hashes in final.json. Native diagnosis
PIDs14025/13356 and head PIDs15056/14338, scope PIDs16906/16124 all exit0. Logs:
`evidence/distributed-d20-{r1,r2,r3}-controller.log`; per-phase rank.json/gpu.log/
final.json/kernel-before+after/guard state+ACK retained.

Six guard records verify clean agents, unchanged per-phase kernel logs and
registries, ACK nonce/phase/boot agreement and source hashes. Archives additionally
include D09 native identity and consumer package manifest; verifier does not
rerun GPU arithmetic. Shared runtime locks/4 CPUs/10GiB RAM/12GiB memory+swap/
512PIDs/256MiB shm/600s watchdog/460s child unchanged. Boots d83af4e1(.104r2)/
bad537ed(.105r3), no development reboot. No jobs/containers/armed guards remain.
Both hosts' docs and agent-memory snapshots synchronized at handoff.

Workstation `kernel/artifacts/distributed-d20` archives:

- .104 final r3:298287 bytes, SHA256 `65824728c50f4e2080207a456996ada8ba286741c0ff28356a600284e261c495`.
- .105 final r3:280711 bytes, SHA256 `54fc6c50af6974c4bc090838aee98d22f7b7479685afc9969c12d9e47e3fc0f9`.

**Next D21 (planned):** practical distributed coverage review and pinned actual
Trainer FSDP FP32 readiness audit. Prioritize a bounded real training integration
with CPU gradient/optimizer and fixed-world restart references if the consumer
API admits it. D08 custom FSDP and D12–D16 actual Trainer DDP are already qualified;
do not repeat their full matrices or infer larger-model capacity/scaling.
