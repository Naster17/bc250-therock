# BC-250 initial Torch performance comparison

Measured on 2026-10-13. These are short GPU benchmarks of the deployed stacks
on two physical boards. They are not clock-normalized measurements of ROCm
alone. No kernel, clock, fan, service or installed runtime changes were made.

## Hosts and controls

| Setting                     | `.104`, new stack              | `.105`, reference stack        |
| --------------------------- | ------------------------------ | ------------------------------ |
| Distribution/profile        | TheRock 10.2 expanded gfx1013  | ROCm 7.2.4 gfx1013 r1.4        |
| Torch                       | 2.13.0a0+gfx1013               | 2.9.1a0                        |
| HIP component               | 7.17.0                         | 7.2.53211-97f5574fe2           |
| Kernel                      | 7.1.5-bc250rocm-r2             | 7.1.5-bc250rocm-r2             |
| GPU                         | Native gfx1013, 20 WGP / 40 CU | Native gfx1013, 20 WGP / 40 CU |
| CPU visibility              | 16 logical CPUs                | 16 logical CPUs                |
| Torch benchmark CPU threads | 4                              | 4                              |
| GPU advertised memory       | 15,938,355,200 bytes           | 15,938,355,200 bytes           |
| User                        | UID/GID 1000                   | UID/GID 1000                   |
| SDMA                        | Disabled                       | Disabled                       |
| Architecture override       | None                           | None                           |
| GPU advertised maximum      | 2,000 MHz                      | 2,230 MHz                      |

Observed temperatures also differed: the reference board reached 91 °C during
these cases, versus 81 °C on the new board. Under load, `freq1_input` reported
implausible MHz values; clock-normalized comparisons are therefore unavailable.
Raw clock, temperature and power readings are retained for investigation rather
than treated as reliable clock measurements. Small percentage differences
cannot be attributed solely to the ROCm port.

The reference host had no cgroup filesystem mounted after reboot. Its existing
runtime image was run in a private mount namespace with a temporary cgroup2
mount. That mount disappeared when the benchmark exited; the host namespace and
persistent configuration remained unchanged. Existing containers were not
recreated, migrated or upgraded. The only persistent new data on `.105` is the
isolated benchmark directory, inputs, cache and results.

## Method and input audit

The script validates GEMM against FP64 CPU accumulation and convolution outputs
and gradients against CPU execution before timing. Copy contents are checked.
Measurements use synchronization around batches, monotonic wall time, and GPU
events. This follows the [PyTorch timing guidance](https://docs.pytorch.org/docs/stable/notes/cuda).

The final comparison uses 400 ms of sustained warmup and five samples targeted
at 200 ms each. Timings below are median per invocation. CNN cases include
forward, loss and backward, with gradient clearing; they do not include an
optimizer step. Timing excludes CPU references and input transfers.

An initial seed-based run revealed different input bytes between Torch versions.
Those diagnostic runs are excluded from the comparison. A single serialized
CPU tensor fixture was generated on `.104` and copied unchanged to `.105`.
All nine case input hashes match. `compare_performance.py` rejects mismatched
inputs/settings or failed correctness checks and flags IQR above 5%.

The final source records invocation settings directly. The archived measured
reports predate that metadata field addition: their normalized copies append
settings from the recorded commands. Unmodified raw reports are retained too.
The synthetic copy-case FLOP field from that earlier metadata bug is omitted
from normalized data; copy bandwidth uses byte traffic, not FLOPs. No timing
samples were altered.

## Matched-input GPU results

Lower milliseconds is faster. Ratio is old time divided by new time.

| Workload                                     | `.104` ms | `.105` ms | Observed new-stack ratio | Interpretation                                    |
| -------------------------------------------- | --------: | --------: | -----------------------: | ------------------------------------------------- |
| GEMM 1024 FP32                               |    0.4345 |    0.4395 |                   1.011× | Essentially similar                               |
| GEMM 1024 FP16                               |    0.3999 |    0.4066 |                   1.017× | Essentially similar                               |
| GEMM 1024 BF16                               |    0.4803 |    0.5080 |                   1.058× | About 5.8% higher throughput in this run          |
| GEMM 2048 FP32                               |    2.9356 |    3.0408 |                   1.036× | About 3.6% higher throughput in this run          |
| GEMM 2048 FP16                               |    3.1071 |    3.1572 |                   1.016× | Essentially similar                               |
| GEMM 2048 BF16                               |    3.6340 |    3.8377 |                   1.056× | About 5.6% higher throughput in this run          |
| Two-convolution forward/backward, 1×16×32×32 |    0.5399 |    1.8447 |            3.417× median | Reference varies heavily; not a stable 3.4× claim |
| Two-convolution forward/backward, 8×32×32×32 |    0.7464 |    0.7454 |                   0.999× | Essentially tied; new samples also show variation |
| 128 MiB device copy                          |    0.6695 |    0.6706 |                   1.002× | Essentially tied                                  |

The small CNN reference samples ranged from **0.6088 to 1.9721 ms** (59.9% IQR),
while `.104` ranged from **0.5386 to 0.5444 ms**. Every measured new sample was
quicker, but the magnitude needs separate investigation. The large CNN case's
new IQR was 5.3%, so its near-zero difference is not a meaningful speedup.

The 2048 FP32 GEMM corresponds to about 5.85 TFLOP/s on `.104`. The copy case
corresponds to about 200.5 GB/s of copied payload, or 401 GB/s when both read and
write traffic are counted. Those are measured operation rates, not theoretical
hardware capability claims.

## SmolLM2-135M training-step performance on `.104`

These timings use the qualified model/adapters, batch 1, sequence length 32,
mathematical SDPA, AdamW and gradient clipping. Base weights are FP32. AMP cases
use FP16 or BF16 autocast with FP32 adapters. Learning rate is zero to keep
weights fixed during timing; backward, clipping and optimizer-state updates
still execute. Loss remains finite and constant, and AMP does not skip updates.

Each case has a first setup step, three warmup steps and five batches of three
timed updates. Reported tokens/s counts 32 processed input tokens per update;
it is training throughput, not generated inference tokens/s. Scheduler, data
loading, logging and checkpoint-file writing are outside the timed region.

| Profile                                     | Median step | Processed tokens/s | Peak Torch GPU allocation |
| ------------------------------------------- | ----------: | -----------------: | ------------------------: |
| Full FP32 fine-tuning                       |   131.56 ms |              243.2 |                 2.609 GiB |
| FP16 LoRA, activation checkpointing enabled |   259.91 ms |              123.1 |                 0.649 GiB |
| BF16 LoRA, activation checkpointing enabled |   262.52 ms |              121.9 |                 0.649 GiB |
| FP32 LoRA, checkpointing disabled           |   111.72 ms |              286.4 |                 0.624 GiB |
| FP16 LoRA, checkpointing disabled           |   138.87 ms |              230.4 |                 0.866 GiB |
| BF16 LoRA, checkpointing disabled           |   140.54 ms |              227.7 |                 0.866 GiB |

The checkpoint-disabled cases assert that recomputation is actually disabled.
They use the same trained FP16 adapter to compare precision profiles. The earlier
checkpoint-enabled cases use their corresponding qualified adapters, so the
checkpoint on/off pair is closest-controlled for FP16. All listed step IQRs are
below 2%.

For this small workload, FP32 LoRA without checkpointing is the fastest tested
profile. Turning off activation checkpointing nearly halves AMP LoRA step time,
at the cost of more activation memory. FP16/BF16 do not automatically accelerate
this board's scalar backends, and autocasting an FP32 base adds conversion work.
This finding should not be generalized to larger models, sequence lengths or
BF16-resident base weights, which have not been benchmarked here.

No production training defaults were changed. The earlier activation-checkpointed
qualification remains valid; the extra cases establish a useful alternative for
this small model when memory permits.

## Files and reproduction

Results are committed under [bc250_performance_results/](bc250_performance_results/):

- `new-104-raw.json`, `old-105-raw.json`: unmodified matched-run results.
- `new-104.json`, `old-105.json`: results with audited invocation metadata.
- `comparison.json`: validated ratios and variability flags.
- `model-checkpoint.json`, `model-no-checkpoint.json`: actual model timings.

Host artifacts are under `/home/nik/bc250work/rocm-performance-20261013` on both
boards. `inputs.pt` is intentionally not committed; its hash is recorded in the
comparison. Benchmark source is in `build_tools/bc250/benchmark_torch.py` and
`benchmark_model_training.py`.

On `.104`, the existing shared-fixture microbenchmark can be repeated with:

```bash
bc250-torch ~/bc250work/rocm-performance-20261013/benchmark_torch.py \
  --fixtures ~/bc250work/rocm-performance-20261013/inputs.pt \
  --label new-104 --sample-ms 200 \
  --output ~/bc250work/rocm-performance-20261013/repeated.json
```

Compare saved reports locally:

```bash
python3 build_tools/bc250/compare_performance.py \
  --new docs/development/bc250_performance_results/new-104.json \
  --old docs/development/bc250_performance_results/old-105.json \
  --output /tmp/bc250-comparison.json
```

Further performance work includes the small-CNN variance investigation, reliable
clock telemetry, FFT/sparse/reduction benchmarks, larger training shapes,
BF16-resident models, compilation overhead and tuning databases. This initial
comparison establishes measured baselines; it does not complete all performance
tasks in [bc250_full_support_roadmap.md](bc250_full_support_roadmap.md).
