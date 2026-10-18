# N2.15: RNG state, offsets and streams — 2026-10-18

Bounded qualification passed on `.104`, native `gfx1013:xnack-`. No RNG arithmetic
defect was reproduced. A stale LFSR113 API comment was corrected in source;
installed binaries, headers, defaults and prefix metadata remain unchanged.
No performance or statistical-randomness certification is claimed.

TheRock tests/controllers `0e4557ee5`; ROCm libraries documentation commit
`c29021b40e758abd3fa97e3c8bddb7f091dd29d2` on native revision
`319a235d6c6496f0da37be44dfa60eaf31a29f82`. Native Torch remains `dc8fea0`,
source/test revision `2b449c4`, mounted from `WS/pytorch-sparse-n213`.
WS is `/home/nik/bc250work/rocm-10.2-gfx1013` on `.104`.

| Passing gate | Fixtures | Evidence |
| --- | ---: | --- |
| Torch state/layout and empty/odd/tail sizes | 80 | [Results](rng-n215-state/result.json) |
| Torch analytic distribution probes | 22 | [Results](rng-n215-statistics/result.json) |
| Torch offsets, isolation, streams, dropout/trace | 20 | [Results](rng-n215-contracts/result.json) |
| Fresh-process mixed RNG continuation | 1 | [Results](rng-n215-resume/result.json) |
| Direct rocRAND/hipRAND | 68 | [Results](rng-n215-native-r2/result.json) |
| Retained P2 RNG | 28 | [Results](rng-n215-legacy/result.json) |
| Retained core/consumer regressions | 7 | [Results](rng-n215-core/result.json) |

## Public Torch scope

State tests exercise FP16/BF16/FP32/FP64 uniform, normal, Bernoulli and exponential,
INT32/INT64 integers, and complex64/128 normal, each in contiguous, step-strided
and transposed output views with nonzero offsets. Exact replay includes final
generator state, finite outputs and untouched exterior/stride-gap sentinels on a
private stream. Seeds above2^63 and offsets above2^32 are used. Twenty boundary
fixtures cover lengths0/1/3/257/65537 for FP32/64 uniform/normal; empty output
does not advance state.

Moment/range probes use65,536 values per fixture, plus separate low/high
parameters for FP32/64 Poisson and gamma. All bounds are fixed in the test and
recorded in the results; low-precision quantization is explicitly allowed.
These are bounded sanity checks, not exhaustive distribution/tail/independence
testing. RNG algorithms differ between CPU and GPU, so their bitstreams are
not compared as interchangeable references.

Torch offset replay covers0/4/256/(2^32+12)/(2^40+4), uniform/normal/INT64.
Offsets advance in multiples of4; `manual_seed` resets them. Three unaligned
offsets and three malformed states are rejected, while legacy seed-only bytes
remain accepted with zero offset. Explicit draws leave the default generator
alone; `fork_rng` restores CPU/GPU states after normal and exceptional exit.
Four host-ordered submissions across two private streams reproduce default-stream
outputs and final state. This does not qualify unordered concurrent mutation.

Two successive40-bit integer draws produce131,074 unique values on this fixture;
the fixed non-reuse threshold is131,000. A private-stream native dropout test
compares output and backward against CPU mask/scale references and verifies
exact RNG replay. The [compressed trace](rng-trace.json.gz) contains five GPU
kernels, including fused dropout, masked backward and Philox normal sampling.

Two fresh worker processes save/restore explicit/default GPU, CPU Torch and
Python RNG states. A mixed continuation matches ten tensor/state fields exactly,
including40-bit integers and a CPU normal state after an odd-length draw.
All checkpoint tensors load on CPU with `weights_only=True`; no user training
script, model, adapter, checkpoint or data file is modified. This complements
the earlier model/optimizer/AMP resume evidence, rather than repeating it.

## Direct native scope and corrected contract

Five pseudo generators (XORWOW, MRG32K3A, Philox, MRG31K3P, LFSR113) pass
uniform/normal/lognormal/Poisson probes at offsets17 and2^32+17. Each resets the
same generator and rebinds a private stream after completed work; output
replay is exact and exterior buffers are guarded. Four Sobol variants, dimensions
1/3 and offsets0/17, match per-dimension slices from longer zero-offset output
and reset replay. MTGP32 correctly returns TYPE_ERROR for offsets and remains
usable after refusal.

A scalar integer CPU Philox4x32-10 oracle compares257 exact words at seven
rocRAND offsets and three hipRAND offsets, including unaligned and >32-bit
offsets. Legacy ordering is set explicitly; raw native offsets are not treated
as Torch's per-thread offsets. The pinned HIP enum is405 for Philox, rocRAND404.
No cross-backend ordering equivalence is claimed.

Initial native run passed59/60: the failed assertion expected LFSR113 TYPE_ERROR
because `rocrand.h` said offsets were unsupported. The library returned success.
Source `rng/lfsr113.hpp` deliberately initializes engines from `m_offset`, and
`test/internal/test_rocrand_lfsr113_prng.cpp` already instantiates offset tests.
The header was stale; disabling this working feature would be incorrect.
Corrected C API comments now identify only MTGP32 as unsupported; the C++ comment
clarifies that its wrapper does not expose offsets. No ABI/implementation change.
Eight additional LFSR distribution/rebinding cases plus an exact legacy-prefix
and non-aliasing high-offset case pass. The initial result/helper/log are retained.

Patch0032 records the source-only documentation change and passes reverse apply
checking. Both source headers were transferred under the exclusive lock after
verifying original hashes and saving exact copies to
`WS/recovery/rng-n215-source-before`. [Header source manifest](header-source.json)
records both revisions and before/after hashes. Installed headers intentionally
retain their original bytes until a future normal build/install; no rebuild is
needed for this comment correction. Native libraries still correspond to319a235.

## Execution and preservation

Controller commands run under `WS/helpers/update_dev_prefix.py -- python3
WS/helpers/qualify_rng_n215_on_host.py`; exact container/subprocess commands are
beside results and in [final.json](rng-n215-qualification-r2/final.json).
The final controller retains the four unchanged successful public gates from
the first attempt (source hashes checked), runs the revised native gate and
remaining regressions, then seals preservation. It does not replay old labels.
No native build/ccache reset/concurrency change or cache deletion was required.

Seven passing GPU container processes plus the initial native assertion failure
have eight acknowledged600s guards; each container deadline is540s. Fresh workers
also have100s deadlines. [Guard states](guard-states.json) and
[kernel digests](gate-summary.json) confirm unchanged logs across the whole batch.
All153 protected identities match, including original/current binaries, runtime
JSON, FFT packs/four persistent AOT databases, old source-header recovery and new
source comments. Current/previous/Torch/ROCm links, image/kernel/boot/defaults,
.105/BIOS/r2 fallback remain unchanged. Free~7.21GiB; no running agent job/guard/
container. Baseline for the next batch: `WS/logs/rng-n215-qualification-r2/final.json`.
Raw logs: `WS/logs/rng-n215-*`; local ignored `kernel/artifacts/rng-n215`.

The historical unresolved CPU Python segfault remains distinct from the fixed
VM-heap/CSR corruption and the explicit API-comment assertion above.

Unqualified: all generator/ordering combinations, offsets near64-bit exhaustion,
large statistical batteries/tails, multi-GPU, threads/process sharing, graph/
compiled RNG equivalence and cross-version bitstreams. Existing checkpoint
exactness still requires matching operations, shapes, types and execution order.

Next N2.16: audit P2 rocPRIM42/hipCUB48/Thrust48, then bounded floating and
segmented reductions/scans with CPU references, guards and private streams.
Use targeted ccache builds; preserve existing primitives and all runtime caches.
