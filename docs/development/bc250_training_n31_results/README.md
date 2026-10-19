# N3.1 current Qwen3.5 training attribution — 2026-10-19

The practical [N2 gap review](../bc250_n2_practical_review.md) found no recorded
mandatory N2 implementation blocking the current training path. Corrected the
support matrix's stale N2.14 cursor and fixed-A01 group1-convolution boundary.
Broad N2 remains open; optional BF16 FFT/sparse solve/CK backward/plugin/RPP and
other workloads retain explicit boundaries. A proposed N2.19 was not executed.

TheRock code `4e76c2f01` supplies an explicit synthetic training profiler, an
offline kernel analyzer and guarded host orchestration. No Torch/library/driver
implementation, application trainer or runtime default changes. Actual local
Qwen3.5-2B base weights are read through the existing qualified model loader,
new nonzero synthetic rank16 adapters and synthetic input IDs only. User training
data/checkpoints are not opened or modified. Actual application helper
`replica/bc250_ops.py` is loaded read-only, with native frozen convolution,
chunk64 FP32 loss and precision-safe gated-delta active.

## Numerical and execution checks

BF16 base/autocast with FP32 adapters, math attention, batch1, checkpointing,
accumulation1. One full warmup initializes optimizer state. Restore owned CPU
copies of372 adapter tensors, optimizer state and CPU/GPU RNG before the paired
capture. Profiled/unprofiled loss, all372 nonzero clipped adapter gradients,
adapter updates, full AdamW state and CPU/GPU RNG match **exactly** at both65 and
1024 tokens. Reference is the unchanged qualified GPU implementation, not a new
whole-model CPU reference. No higher-order/compile/graph/convergence claim.

Each capture executes36 causal-convolution and36 gated-delta calls:18 layers
in forward plus checkpoint recomputation, with one chunked-loss call. Each final
shape then completes three warm unprofiled finite full training updates. Four
offline tests cover GPU annotation exclusion, autograd worker-thread launches,
interval overlap versus sum, input-shape attribution, missing correlations,
ambiguous stages and nonpositive durations.

## Measured current-boot baseline

| Synthetic input length | Kernel records | Median warm whole-step wall | Peak allocated in warm samples |
| ---------------------- | -------------- | --------------------------- | ------------------------------ |
| 1024                   | 38314          | 8.676s                      | 5351892480 bytes               |
| 65                     | 17286          | 0.726s                      | 4232892416 bytes               |

1024-token capture:8.680s summed kernel duration, **84.71%** attributed to
`aten::mm` launches. Backward-submitted kernels account for76.82% of kernel
sum; clip and optimizer together about0.046%. These are kernel-duration shares,
not a universal model profile or independent stage speedups.

| BF16 mm inputs in1024-token capture | Share of kernel-duration sum |
| ----------------------------------- | ---------------------------- |
| [1024,6144] @ [6144,2048]           | 45.94%                       |
| [1024,2048] @ [2048,6144]           | 16.53%                       |
| [1024,2048] @ [2048,2048]           | 4.46%                        |
| [1024,2048] @ [2048,248320]         | 4.20%                        |
| [1024,248320] @ [248320,2048]       | 4.09%                        |

The first shape has132 kernel dispatches, not necessarily132 mathematical mm
calls: one operation can launch several kernels. Shape metadata does not encode
all strides/transposes/alignment; inspect these before kernel selection.

The65-token control also spends87.66% of kernel sum in mm, with different shape
costs. Native triangular solves are only~0.58% in the initial1024 trace, so
adding general sparse solve or more tiny eig coverage would not address this
measured bottleneck. This does not quantify all gated-delta costs or Amdahl
speedup for any future fusion.

Profile wall with setup/teardown is11.446s at1024; record-shape collection adds
cost. It is separate from8.676s warm unprofiled timing. Three warm samples are
not sustained user throughput: no data loading/evaluation/checkpoint saving,
accumulation16, real labels, long runs or thermal sweep. No speed improvement
is claimed by this batch.

## Collector correction and retained attempts

- r1 numerical replay/profile passed. Initial summary mixed CPU operation,
  device kernel and GPU annotation views. In particular, the main-thread backward
  annotation did not aggregate all worker-thread GPU work. Raw trace was intact.
- Added offline `analyze_training_n31.py`: use `cat=kernel` only; correlate each
  launch with its CPU runtime submission, partition into sequential stage spans
  across CPU threads, and map `External id` to operation/input shapes. GPU
  annotation ranges are excluded. Duration sum and per-device interval union are
  separate metrics. This corrects this new reporting tool, not an A09 runtime bug.
- r2 test controller failed before GPU launch with an undefined leftover prefix
  variable; corrected. Error/source preserved, no guard was armed.
- r3 final1024 capture and r4 short65 control pass. r4 adds the explicit length
  CLI and removes an unused import; owned math/optimizer operations are unchanged.
  Exact r3 sources are archived with their result hashes.

## Recovery, preservation and reproduction

Run through the existing exclusive lock; the launcher receives the explicit
running retained r2/r3 guard release. Example with a fresh attempt directory:

```sh
python3 WS/helpers/update_dev_prefix.py -- \
  python3 WS/helpers/qualify_training_n31_on_host.py --attempt r5 --length 1024
```

No native build or ccache change is needed. Current libraries and four FFT AOT
DBs, N2 binaries/headers, runtime metadata/links and user trainer/config source
are covered by193 matching protected identities. Three GPU processes all pass
with600s guard ACK; dmesg identical across all four attempts. Same entry kernel
7.1.5-bc250rocm-r2 / boot d0532075-88fb-4a2f-974e-f61578d20d9f, no reboot or boot
policy change. Native Torchdc8fea0/source2b449c4, native libraries319a235d6/source-
doc c29021b40e7, BIOS/.105/SDMA0/heap-off/math/default rocBLAS preserved.
No active job/container/guard; free~6.85GiB at qualification end. Historical
Python CPU segfault remains unresolved/separate from fixed VM-heap/CSR.

[Final1024 result](training-n31-r3-profile/result.json),
[short control](training-n31-r4-profile/result.json),
[final preservation](training-n31-qualification-r4/final.json) and
[guard/trace hashes](gate-summary.json). Full traces retained on .104 at
`WS/logs/training-n31-r{1,3,4}-profile/training-step-trace.json`; exact local
ignored copies in `kernel/artifacts/training-n31/`. Raw traces are65–152MB and
stay outside Git. Reanalyze offline with:

```sh
python3 build_tools/bc250/analyze_training_n31.py \
  --trace /path/to/training-step-trace.json --output /path/to/analysis.json
```

Next **N3.2**: audit rocBLAS/Tensile selection and actual strides/transposes for
the dominant BF16 [1024,6144]@[6144,2048] shape, compare one hardware-feasible
scalar candidate with fixed CPU/qualified-GPU references and paired timings,
then check model gradients before any reversible deployment. Keep default
rocBLAS/math routing until demonstrated correctness and workload benefit.
