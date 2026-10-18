# N2.14: FFT plans, layouts and backward — 2026-10-18

The installed native gfx1013 runtime passes the bounded extension below. No
production FFT defect was reproduced; no Torch/ROCm binary, dispatch, default,
prefix metadata or link changed. This is additional qualification, with no
performance claim. BF16 promotion B01 remains separate.

TheRock qualification code: `44568c2e2`. Active Torch native revision
`dc8fea0cb8b157cb415359033e4f0175d2ebfa56`, source/test revision `2b449c4`,
ROCm libraries `319a235d6c6496f0da37be44dfa60eaf31a29f82`. Active Torch remains
`WS/pytorch-sparse-n213`; WS is `/home/nik/bc250work/rocm-10.2-gfx1013` on `.104`.

| Gate | Passing fixtures | Evidence |
| --- | ---: | --- |
| Complex FFT/IFFT | 96 | [Results](fft-n214-r2-complex/result.json) |
| Real FFT / independently supplied inverse spectra | 72 | [Results](fft-n214-r2-real/result.json) |
| Explicit DFT, finite differences, plan reuse / GPU trace | 3 | [Results](fft-n214-r2-contracts/result.json) |
| Direct hipFFT plan-many | 16 plans / 32 executions | [Results](fft-n214-r2-native/result.json) |
| Retained P2 FFT | 30 | [Results](fft-n214-r2-legacy/result.json) |
| Retained core output/backward/RNG/consumer groups | 7 | [Results](fft-n214-r2-core/result.json) |

The 168 public fixtures each check forward output, weighted-loss backward into
the original leaf (including unused storage), strided `out=` with all sentinel
gaps preserved, and exact input immutability on a private GPU stream. References
are CPU float64/complex128, using the same originally rounded inputs. Real and
imaginary loss weights plus a quadratic term avoid a phase-insensitive energy
test. Fixed tolerances are `rtol=atol=5e-4` for FP32/C64, `1e-10` for FP64/C128.
Maximum public output errors: complex `6.39e-6 / 1.51e-14`, real/inverse
`5.61e-6 / 1.43e-14` for single/double precision. Maximum leaf gradient errors
are `5.80e-8 / 1.12e-16` across both groups. These maxima describe these inputs.

Cases include three normalizations, batched prime/composite sizes, step slices
with nonzero offsets, transposes, expanded zero-stride batches, lazy conjugation,
crop/pad, nonadjacent/reordered dimensions and a four-transformed-dimension
decomposition. Odd real lengths include 17,45 and multidimensional 7x13;
inverse spectra are random inputs, not only paired roundtrips. An explicit
17-point DFT differs by `3.29e-14`; three central finite differences for odd RFFT
backward differ by at most `5.50e-11`.

The direct wrapper test calls the pinned AMD pointer-handle ABI, six
single/double real/complex APIs, lengths17/32, batch3, input/output strides2/3,
nonzero pointer offsets and padded batch distances. Automatic workspace
allocation is disabled; the returned size is allocated and bound explicitly.
Each plan executes twice with fresh inputs, rebinding to a second private stream
after synchronization, then synchronizes before destruction. CPU comparisons
account for unnormalized native inverses. Maximum direct output error is
`1.75e-6`; outer buffer guards survive all executions.

## Existing implementation and important contracts

Audit used qualified Torch `SpectralOps.cpp`, `CuFFTPlanCache.h`, existing
`test_spectral_ops.py`, P2 FFT30, and the pinned hipFFT header/rocFFT layout docs.
Torch already clones ROCm FFT inputs; C2R has an additional preservation path,
plans bind the current stream/workspace each execution, and transforms above
three dimensions are decomposed. No duplicate backend or replacement was added.

Native rocFFT explicitly permits temporary writes into gaps of noncontiguous
buffers and into input buffers, including out-of-place C2R. See
`rocm-libraries/projects/rocfft/docs/how-to/working-with-rocfft.rst`,
“Overwriting non-contiguous buffers.” The direct test therefore protects the
outside of the full reserved buffer spans and records internal output gap writes
(zero on these fixtures); it does not impose Torch's stronger public `out=`
contract on raw hipFFT. No input-preservation promise is made for raw callers.

The process-local Torch plan cache grows `2 -> 3 -> 3` for the same transform
reused on two streams; capacity1023 is unchanged, with no cache clear/resize.
The trace contains34 GPU kernels including named native rocFFT forward/backward
length17 kernels: [compressed Chrome trace](fft-trace.json.gz). Persistent AOT
databases are separate from this process-local plan cache.

## Execution, preservation and retained failure

Three Python helpers were staged from the local workspace. No native build was
necessary, so ccache/build concurrency settings and all build products remain
unchanged. Exact six subprocess commands are in
[final.json](fft-n214-qualification-r2/final.json); container commands and logs
are beside each result. Controller command, executed in a root SSH PTY:

```sh
python3 WS/helpers/update_dev_prefix.py -- \
  python3 WS/helpers/qualify_fft_n214_on_host.py
```

The existing exclusive lock covers the controller. Each GPU gate uses its own
540s container deadline and600s boot guard; all seven guards, including the
initial failed test process, are acknowledged: [states](guard-states.json).
All kernel logs are byte-identical, including across that failed process;
[gate digests](gate-summary.json) and compressed initial kernel log retained.
The final controller verifies145 protected identities, including prior recovery
cohorts, current Torch/ROCm DSOs, runtime JSON, FFT packs and all four existing
rocFFT build/stage/dist AOT databases. Active/current/previous/ROCm/Torch links,
kernel/boot, BIOS, .105 and r2 fallback are unchanged. No installation/reboot,
cache deletion or user-file changes. Free space about7.22GiB; no running agent
job/container/guard. Latest preservation baseline is the final JSON above.

Initial complex run had48 passes/48 test setup failures: for complex128 the CPU
reference leaf reused the original tensor, so copying it to the GPU created a
non-leaf and `.grad` was absent. The corrected fixture detaches the GPU copy
before `requires_grad_` and asserts both leaves. No tolerance, production code
or expected output changed. [Initial results](fft-n214-complex/result.json),
initial helper and controller logs are retained compressed. This was a Python
test exception, not the historical unresolved CPU Python segfault or the already
fixed VM-heap/CSR corruption.

## Limits and next action

No in-place native plans, callbacks, planar complex layouts, giant/64-bit plans,
multi-GPU, simultaneous cross-thread plan use, higher-order gradients,
compile/graphs, FP16 expansion, BF16 promotion or broad speed/stability claim.

Next N2.15: audit retained P2 RNG28 and existing checkpoint/state tests, then
qualify bounded generator offset/state replay and private-stream sequencing
through Torch/rocRAND against documented exact replay and analytic distribution
contracts. Do not compare CPU/GPU RNG bitstreams as if they share an algorithm.
Fix only demonstrated defects; preserve existing scientific paths and caches.
