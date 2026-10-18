# N2.16: floating, custom and segmented primitives — 2026-10-18

The existing installed headers pass 1,536 new native GPU checks on `.104`:
rocPRIM 576, hipCUB 480 and rocThrust 480. Retained P2 INT32 primitives 138 and
Torch core7 also pass. No production primitive defect was reproduced; no
library/Torch/header installation, dispatch or default change was needed.
This adds qualification, not a new Torch operation or a measured speedup.

Qualification source: TheRock `2db3b18c1`. Native Torch `dc8fea0`, source tests
`2b449c4`, native ROCm libraries `319a235d6`; source-only RNG documentation
revision `c29021b40e7` is retained. WS denotes
`/home/nik/bc250work/rocm-10.2-gfx1013` on `.104`.

| Native gate | Checks | Evidence |
| --- | ---: | --- |
| rocPRIM FP32/FP64 and custom argmax | 576 | [Results](primitives-n216-r1-rocprim/result.json) |
| hipCUB FP32/FP64 and custom argmax | 480 | [Results](primitives-n216-r1-hipcub/result.json) |
| rocThrust FP32/FP64 and custom argmax | 480 | [Results](primitives-n216-r1-thrust/result.json) |
| Retained P2 INT32 | 42 +48 +48 | [Results](primitives-n216-r1-legacy/result.json) |
| Torch core output/gradient/RNG/consumers | 7 | [Results](primitives-n216-r1-core/result.json) |

## Scope and references

Audited the existing P2 sources, pinned primitive headers and installed API
signatures before adding tests. P2 covered INT32 reductions/scans/sorts/selection,
block/wave32 operations and explicit Thrust HIP policies; it was retained intact.
New standalone tests use those installed headers, not another upstream checkout.

Each backend covers lengths 0/1/33/257/4099/65537 on default and nonblocking
private HIP streams. FP32/FP64 cases include ordinary reduction with nonzero init,
inclusive/exclusive scans, scan-by-key, segmented reduction, in-place inclusive
and in-place exclusive-by-key scans. Keys repeat nonadjacently, while adjacent
runs stay distinct. Boundaries use uneven lengths 1/31/32/33/255/257/4099, internal
empty segments, and empty first/last segments. Entirely empty input is included.
rocPRIM additionally exercises its offset-based inclusive/exclusive scan APIs.

Two floating patterns are tested. Dyadic inputs (bounded integer multiples of
1/64) require exact agreement. Non-dyadic rounded inputs use independent CPU
long-double accumulation, fixed bounds `5e-4 + 4e-5*abs(reference)` for FP32 and
`3e-11 + 5e-12*abs(reference)` for FP64. The same bounds apply to every size;
they were not adjusted after testing. Maximum observed errors across all three
backends are `5.72e-6` FP32 and `8.77e-15` FP64. These well-scaled fixtures do not
establish general conditioning or bitwise floating-point reproducibility.

Custom FP32/FP64 value/index aggregates exercise reduction, inclusive scan,
scan-by-key and segmented reduction. The associative argmax operator chooses the
first global index on ties, with repeated maxima and an explicit empty identity.
CPU references update the best value/index directly rather than invoking the
device functor. All 288 custom checks are exact.

Read-only inputs, keys and offsets are checked byte-for-byte after use. Typed buffers
have 256-byte prefix/suffix guards and nonzero data offsets; unused output tails
are protected too. rocPRIM/hipCUB query temporary-storage size, allocate aligned
guarded workspace and check its boundaries after execution; maximum queried
workspace is 7,196 bytes; Thrust's internal temporary allocations are not measured
by that counter. Every result is synchronized and copied for CPU checking.
Buffer initialization/verification also synchronizes, so this is stream binding
qualification, not overlapping-concurrency or timing evidence.

## Different public API contracts

rocPRIM/hipCUB offset-based segmented reductions include empty segments, which
must produce the supplied initial value. Thrust `reduce_by_key` returns only
nonempty contiguous key runs and has no reduction-init argument. Its returned
keys/count and values are checked against that contract; no empty-segment
emulation is inserted. Thrust's ordinary reduction computes through explicit HIP
policy and returns a host scalar; the adapter copies it to a guarded output for
the shared checker. It is not replaced with a CPU arithmetic implementation.

Thrust uses `thrust::hip::par_nosync.on(stream)` explicitly, rather than implicit
dispatch through a host iterator. rocPRIM/hipCUB device APIs receive HIP device
pointers and the selected stream. Every executable checks actual native gfx1013
and wave32 at startup. No architecture spoofing is used.

## Build, execution and preservation

Only three test executables were built, one from each compile-time backend of
`build_tools/bc250/primitives_n216.cpp`; common guards/oracles are in
`primitives_n216_common.hpp`. Compiler: installed `amdclang++ -x hip -O2
-std=c++17 --offload-arch=gfx1013`. Builder image
`fa3e8c254b6a791b541d49d4b0a1000a5d18aaa661b41e89794b82c104770238`,
tag `localhost/bc250-rocm-build:10.2-ccache`, without GPU devices/network.
Three parallel compiler jobs stay below the native 16-job limit; OMP4,
`setup_ccache.py --no-reset-stats`, effective `CCACHE_MAXSIZE=20G`. Build log
[ccache evidence](primitives-n216-qualification-r1/build.log) is cumulative,
including historical cache errors, not three new build failures. All three
compiler logs are empty and the first build/qualification attempt passes.

New output directory `WS/build/bc250-coverage/primitives-n216-r1` preserves old
builds and the original P2 binaries. Source/build/prefix mounts are read-only;
only the separate output and existing ccache mounts are writable. Exact build,
test commands, binary/source/protected hashes and final seal:
[final.json](primitives-n216-qualification-r1/final.json). Controller command:

```sh
python3 WS/helpers/update_dev_prefix.py -- \
  python3 WS/helpers/qualify_primitives_n216_on_host.py --attempt r1
```

The exclusive runtime lock covers build and serial GPU gates. Each standalone
binary has a 90s timeout, each GPU container 540s and its boot guard 600s; build 1200s.
Five guards acknowledged, no failed GPU process and kernel logs unchanged for
the entire batch. [Guard states](guard-states.json), [gate/kernel summaries](gate-summary.json)
and compressed initial kernel log retained. All 168 protected identities match:
prior 153, three original P2 binaries and 12 selected installed primitive headers.
Current/previous/ROCm/Torch links, runtime JSON, DSOs, FFT packs/four persistent
AOT databases, kernel/boot/image/.105/BIOS/r2 fallback/defaults are unchanged.
No cache reset or cleanup. Free space is about 7.19 GiB; no running agent job/container/guard.

Next preservation baseline is `WS/logs/primitives-n216-qualification-r1/final.json`,
including its new `binary_sha256` entries if those test products are relevant.
Raw logs `WS/logs/primitives-n216-*`, binaries/build logs in the output directory;
local complete ignored copies `kernel/artifacts/primitives-n216`.
Historical unresolved CPU Python segfault remains separate from fixed VM-heap/
CSR corruption. No crash or new numerical defect was seen here.

Unqualified: NaN/infinite/signed-zero arithmetic policies, low-precision types,
noncommutative custom operators, huge/64-bit sizes, deterministic-scan variants,
floating sorts/selection, graph/compiled integration, overlapping streams,
multi-GPU, broad performance or all Torch primitive consumers/autograd paths.

Next N2.17: audit retained P2 sorting/selection, then bounded floating key/value
stable sorting and selection with explicit ordering/special-value contracts,
CPU references, guarded outputs and private streams; fix demonstrated defects.
