# N2.17: floating stable sorting and selection — 2026-10-18

The installed native gfx1013 stack passes 1,960 new C++ checks and 62 public
Torch fixtures on `.104`. Retained P2 INT32 primitives 138 and core7 also pass.
No production defect was reproduced; installed headers, binaries, dispatch,
defaults and prefix metadata remain unchanged. No training speedup is claimed.

TheRock qualification `5826261d8`; native Torch `dc8fea0`, source/test `2b449c4`,
native ROCm libraries `319a235d6`, source-doc `c29021b40e7` retained.
WS is `/home/nik/bc250work/rocm-10.2-gfx1013` on `.104`.

| Gate | Checks | Evidence |
| --- | ---: | --- |
| rocPRIM stable pairs/selection/contracts | 656 | [Results](sorting-n217-r2-rocprim/result.json) |
| hipCUB stable pairs/selection/contracts | 656 | [Results](sorting-n217-r2-hipcub/result.json) |
| rocThrust stable pairs/selection | 648 | [Results](sorting-n217-r2-thrust/result.json) |
| Torch output/indices/gradient consumers | 62 | [Results](sorting-n217-r2-torch/result.json) |
| Retained P2 INT32 | 42 +48 +48 | [Results](sorting-n217-r2-legacy/result.json) |
| Retained Torch core/consumers | 7 | [Results](sorting-n217-r2-core/result.json) |

## Ordering contracts and CPU references

Audit used P2 sources, pinned rocPRIM `type_traits.hpp` floating radix codec,
radix/merge-sort/select APIs, hipCUB wrappers and Torch sorting/indexing sources.
Native radix sorting is stable and treats signed zeros as equivalent. The pinned
codec orders NaNs by transformed sign/payload bits; negative NaNs can precede
negative infinity. These tests record that implementation contract, not a
cross-version IEEE total-order promise. Nondefault float radix bit ranges are
explicitly unsupported; eight size-query refusals per rocPRIM/hipCUB backend
return `hipErrorInvalidValue` and leave guarded output unchanged.

Comparison sort uses an explicit strict weak comparator: numeric ordering with
all quiet NaNs equivalent and last, in both directions, and signed zeros
equivalent. GPU comparison uses integer encodings; its independent CPU oracle
uses `std::isnan` and numeric comparison. Raw radix cases use a separate scalar
CPU encoding. CPU `std::stable_sort` operates on original payload indices;
returned keys are compared bit-for-bit and indices exactly. Thus ties, zero
signs and quiet-NaN sign/payload preservation are all checked.

rocPRIM uses radix pairs and in-place merge sort; hipCUB uses DeviceRadixSort and
explicit StableSortPairs. Thrust uses explicit HIP `par_nosync` stable-sort
policy: default less/greater for NaN-free inputs, and an explicit radix-rank
comparator for the raw-NaN pattern. Its separate NaNs-last comparator cases are
also tested. This does not qualify default Thrust comparison with NaNs.

## Native coverage

FP32/FP64, lengths 0/1/33/257/4099/65537, default/private streams and three input
patterns cover repeated finite keys, both signed zeros, smallest subnormals,
largest finite values, infinities and positive/negative quiet NaNs with repeated
payloads. Each context checks ascending/descending radix-or-default pairs,
ascending/descending custom stable pairs, then five selection policies:
finite nonnegative (both zeros included), negative sign bit (including negative
zero/NaNs), all, none, and flagged selection using 0/1/2/255 flags.
CPU selection references independently classify values/flags and check exact
counts, stable sequence and output bits. Every context has nine result checks;
the 16 float bit-range refusals give the final total of 1,960.

The unchanged N2.16 buffer/workspace utilities provide 256-byte guards, nonzero
data offsets, exact unused-output-tail checks and bytewise preservation of
read-only inputs, payload indices and flags. Query-allocated temporary storage
has guards too; maximum measured rocPRIM/hipCUB workspace is 786,444 bytes.
Thrust internal allocations are not measured by that counter. Initialization
and verification synchronize, so stream binding is checked without a concurrent
overlap or asynchronous-completion claim.

## Public Torch consumers

54 finite fixtures cover FP32/64 stable sort in both directions and
`masked_select`, lengths 0/33/4099, contiguous/step-strided/transposed inputs,
private streams and untouched input storage. Outputs, sort indices and complete
base-storage gradients match CPU exactly with supplied upstream gradients;
unused storage gradients are included. The sort backward check verifies the
selected-index scatter contract, not differentiability at tied values.

Eight additional strided fixtures compare stable sort values and indices for
NaNs, infinities and signed zeros. They follow Torch CPU semantics (NaNs last
ascending, first descending), independently of the native custom comparator
above. Non-finite fixtures do not claim gradients or bitwise NaN payload
semantics. No Torch source or autograd implementation was changed.

## Builds, retained failure and preservation

Three standalone executables were built with installed `amdclang++ -x hip -O2
-std=c++17 --offload-arch=gfx1013`, ccache20GiB/OMP4/three jobs (native limit16),
`setup_ccache.py --no-reset-stats`, no mapped GPU/network. Build image
`fa3e8c254b6a791b541d49d4b0a1000a5d18aaa661b41e89794b82c104770238`.
Source/build/prefix mounts are read-only; output and the existing cache are
writable. Runtime tests check actual gfx1013/wave32; no architecture spoofing.

Initial attempt r1 failed compiling the new Thrust adapter: one `auto`
declaration combined `device_ptr<T>` and `device_ptr<int>`. Split declarations
fix this test-only error. The [initial compiler log](sorting-n217-qualification-r1/thrust-build.log),
compressed initial source and both controller/build records are preserved.
r1 produced rocPRIM/hipCUB binaries but ran no GPU test. r2 retains separate
outputs at `WS/build/bc250-coverage/sorting-n217-r2`; all three builds and six
GPU processes pass. Neither attempt overwrites earlier builds/binaries/caches.

Controller runs under the existing exclusive prefix lock:

```sh
python3 WS/helpers/update_dev_prefix.py -- \
  python3 WS/helpers/qualify_sorting_n217_on_host.py --attempt r2
```

Exact build/container commands, source/binary hashes and final preservation:
[final.json](sorting-n217-qualification-r2/final.json). Build timeout 1200s;
native binaries 90s, GPU containers 540s and six acknowledged 600s boot guards.
[Guard states](guard-states.json) and [gate/kernel summaries](gate-summary.json)
verify unchanged kernel logs across both attempts. All 181 protected identities
match: prior 168, three N2.16 binaries, nine sorting/selection headers and the
unchanged shared guard helper. Current/previous/ROCm/Torch links, DSOs/runtime
JSON, four persistent FFT AOT databases/packs, kernel/boot/defaults/.105/BIOS/r2
fallback remain unchanged. Free space about 7.16GiB; no running agent job/container/guard.

Next preservation baseline is `WS/logs/sorting-n217-qualification-r2/final.json`
plus its `binary_sha256` when protecting the new test executables. Raw
`WS/logs/sorting-n217-*`, build products in the two separate output directories;
complete local ignored copies `kernel/artifacts/sorting-n217`. Historical Python
CPU segfault remains unresolved/separate from fixed VM-heap/CSR corruption and
the explicit compile error above.

Limits: quiet NaNs only, no signaling-NaN policy, half/BF16, arbitrary types,
segmented sorting, huge/64-bit sizes, all iterators/allocators, overlapping
streams, graph/compiled consumers, cross-version ordering or broad performance.
Public gradients are bounded first-order CPU comparisons, not universal Torch
autograd qualification.

Next N2.18: audit existing CK scalar DL admission/GEMM evidence, then qualify one
hardware-admitted DL convolution/grouped case with CPU references and targeted
builds. Keep instruction-dependent XDL/WMMA and the modern MIOpen CK plugin off;
do not repeat already-qualified CK Tile normalization/top-k work.
