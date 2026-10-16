# A02: native legacy CK reductions — 2026-10-16

Implemented and installed on .104. ROCm libraries `d375d8a2ea7280170455d133058c3cf8e16f58b6`
contains native-profile commit `2093090048f` and the BF16 correction; TheRock
source/helpers `d17427bfeff6ab7204855a753a883c9a66fbf51b`. Torch4ee6571,
ROCr329eaf7, Triton ee65ff3b, image5df0a58, kernel0957f1/image84995f1 and boot
a287fd16 are unchanged.

MIOpen generated `CK_AMD_GPU_GFX1013`, but its bundled legacy CK config rejected
it. Direct API sums and tiny LSTM backward reproduced that failure independently
of optional modern CK plugins. Added the native RDNA1 buffer descriptor0x31014000,
scalar FMA capability, defaults excluding legacy packed-dot inline assembly, and
an explicit native XDLOPS refusal. Compile-only metadata confirms wave32.

The expanded tests also exposed missing BF16 enum mapping/raw-buffer access and
implicit conversion of ushort storage to numeric values. BF16 now uses existing
float conversion helpers and bit-preserving16-bit buffer transfers. Reduction
scalars and alpha/beta epilogues use compute precision, converting only at the
output store. BF16 with non-FP32 compute is explicitly rejected before launching.
This is native GPU arithmetic; CPU double computations serve only as test references.

MIOpen's binary-cache key omits embedded header contents. An intermediate build
reused an incorrect BF16 kernel. Reduction compiler options and in-memory network
keys now carry version2, retaining existing caches while isolating these kernels.

| Gate | Bounded evidence |
| --- | --- |
| Original A01 runtime |Two direct FP32 sums and LSTM backward fail with missing target macro; collector exit0 means collection, not success |
| Initial native candidate |Smoke3/3 and FP32/FP16 LSTM/GRU/tanh-RNN12/12 outputs/all gradients pass |
| Expanded BF16 investigation |Retained missing mapping, raw-buffer compile refusal and wrong-number/stale-cache attempts; guards intact |
| Final candidate, existing cache |69/69:54 numerical reductions,3 explicit non-FP32 BF16 refusals and12 recurrent cases |
| Installed normal MIOpen |Same54 numerical reductions +3 refusals; separate fresh MIOpen cache, existing lower-level compiler caches retained; no preload/library-path override |
| Installed recurrent |12/12 LSTM/GRU/tanh-RNN: FP32/FP16,1-layer unidirectional and2-layer bidirectional; outputs, states, input/state and every parameter gradient match quantized-input CPU-double references |
| Installed regressions |30 convolution configurations,32 stable positive HIP event pairs and seven core GPU/reference groups plus pinned consumer imports |

Reductions cover partial/all dimensions, lengths7/65/513/4097 (thread, wave32,
block, multiblock), FP32/FP16/BF16 storage with FP32 compute, all eight public
operations at length65, non-unit alpha/nonzero beta across methods, one strided
custom-stream fixture and FP32 min/max indices. Output/workspace/index guards
remain intact. Tolerances are FP32 4e-4, FP16 .004, BF16 .03; recurrent FP32 4e-4
and FP16 .008. These small fixtures establish neither universal correctness nor
speedups. BF16 RNN is refused by existing MIOpen RNN code and is not enabled.
NaN/tie/index variants, other compute types, packed sequences, projections,
dropout and large recurrent workloads remain unqualified.

All14 executed GPU gates, including failures, ACKed600s recovery guards. Every
before/after kernel log hashes to
`53f83f7bbee6c060e3153e7ae95a4ff069bf9627113591c39cdddb9ab5f4b973`.
[gates.json](gates.json) retains exits separately from case outcomes.

Targeted16-job builds used existing20GiB ccache without GPU mappings. Deployment
held the exclusive prefix-update lock, installed only MIOpen's runtime component
into separate staging with installed RPATHs, then atomically replaced one DSO.
Automatic rollback to A01 was prepared; success gates passed and rollback was
not needed. Independent binary/source/metadata backups remain in
`WS/recovery/miopen-a02-before`. Installed MIOpen SHA256:
`f4b1a10ce20bdbdf0e6ac90168fb50945b35cf8e26385b66f3ee387aa4507b3c`.
Eight protected HSA/HIP/Torch/BLAS/kpack/frozen-fallback hashes are unchanged;
see [deployment manifest](miopen-a02-deployment/result.json). Source export
component GIT_REVISION updated; superbuild baseline labels retained. Installed
`WS/SOURCE_MIOPEN_A02.json` is also copied into the current prefix.

Applies automatically to callers loading the current development MIOpen,
including bc250-torch/bc250-train, beyond the LoRA application. Other prefixes,
standalone venvs and the frozen fallback are outside this deployment. .105, BIOS,
boot policy/r2 fallback, SDMA0/heap-off/math defaults and user sources/data stay
unchanged. Historical Python CPU fault remains unresolved and distinct from
fixed VM-heap/CSR corruption. A01 clock limits remain; earlier GPU-event speed
ratios still require remeasurement.

All29 library patches reconstruct the exact committed source tree; see
[reconstruction.json](reconstruction.json). The older P2 Torch patch-bundle SDPA
packaging gap is separate; qualified Torch Git/source preserved. Complete logs
remain on .104 `WS/logs/miopen-a02-*` and locally under ignored
`kernel/artifacts/miopen-a02/host-logs`. No agent job/container/deadline remains.

Exact next **A03**: inspect genuine gfx1013_20 FindDb/PerfDb/kernel-cache records,
collect bounded tuning entries with device/version/shape provenance and validate
replay/package scope; review capability-aware optional CK plugin probing. Missing
system database/plugin warnings remain open; never fabricate databases or rename
another architecture's data. A03–A09 precede N2.7; K6 stays completed.
