# B06.7: relocatable opt-in MIGraphX SDK

The SDK archive passes 134 checks across eight guarded gates on .104. Two
independently extracted prefixes, including a path with spaces, run Python ONNX
inference, compiled serialization, explicit HIP copies, a public C API consumer
through its C++ wrapper, and the installed CLI. The original MIGraphX build trees
are absent from the qualified containers and the storage bridge is disconnected.
This is an opt-in archive, not an active prefix/image/SDK registry installation.
The daily Torch/RCCL profiles, defaults, boot and user applications are unchanged;
.105 was not contacted.

## Artifact and use

The workspace artifact is:

`kernel/artifacts/migraphx-b067/bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067.tar.gz`

41,753,676 bytes; SHA256
`11f7f7ab1923132131dc0118af62fbd425e55cfd802d86fdcb57e4952583c8ba`.
It includes 565 regular files and 12 contained library links: Python3.12 extension,
C API/graph libraries, public/generated headers, CMake metadata, CLI and HIPRTC
helpers, source snapshots, license texts, source/fetch pins and owned fixtures.
The package depends on the selected native ROCm/glibc runtime image; it is not a
standalone ROCm distribution or a native musl-host Python package.

Extract into an owned prefix, then use the SDK's child-process launcher:

```sh
SDK=/absolute/path/bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067
bc250-torch --mount-ro "$SDK" "$SDK/bin/bc250-migraphx.py" app.py
```

The SDK launcher prepends its relative Python/library directories and the native
ROCm SQLite directory for its child only. It installs no Python package or loader
configuration. It does not redirect Torch operations or register an ORT GPU EP.
The archive's README covers FP32 parse/compile/run, explicit output allocation,
source/licenses and `${migraphx_LIBRARIES}` for a CMake consumer.

The qualified containers were built with the existing public launcher's command
builder, manifest, runtime lock, devices and native prefixes, using an owned HOME
and a restricted mount set. Ordinary installed-SDK activation/profile integration
is the next admission; no user application was inspected or tested.

## Source repair and installed layout

MIGraphX revision is `4797a61125bbec318005969f8497dbd50849cd6e`, over B06.6's
`622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83`. The only component source change is
`rocm_export_targets(TARGETS migraphx::c)`: the old generated configuration advertised
`migraphx::migraphx_c` while actually exporting `migraphx::c`. The external consumer
reproduces the missing-target CMake failure against sdk-r4. The identical consumer
then configures, compiles, links and executes against the corrected sdk-r5/archive.
The existing exported name and ABI are retained; no arithmetic, dtype admission,
architecture whitelist or backend default changes.

The isolated target build adds the generated C API and CLI to B06.6's Python/GPU
candidate, then uses normal CMake installation into a fresh prefix. Source snapshot
4,782 files matches the new commit; source archive SHA256
`4b8c3929a711396e06251e97fa18d0474e3dbd26cf8c38d89b6a38b298dc1187`.
Fresh reconstruction from the preserved B06.2 full Git bundle, B06.3/B06.4 repair
bundles and this batch's 542-byte repair bundle reaches the exact commit and passes
full fsck. Dependency pins remain private Abseil20250512/Protobuf30/pybind11 2.12.0;
existing Torch/profiler sources and Python protobuf remain unchanged.

The normal install inherited the external build wrapper's broken license link;
packaging materializes that one known file from the verified MIT license source.
Other broken or escaping links are refused. Installed libraries live under
`lib/migraphx/lib`, whereas upstream installs the HIPRTC executable in `bin`.
A byte-identical sibling copy beside the GPU library supplies the first existing
lookup location. A parent-traversing directory-link alternative was unreadable
through SSHFS (EPERM) and is not used. Both qualified roots actually execute the
sibling HIPRTC driver, observed from a separate process while C++ holds the GIL.
No global search-path or backend source workaround was added.

## Executed acceptance

Each prefix passes Python50: the retained six FP32 IR10/opset18 models, Python
file/buffer parsing, shape/type checks, uncompiled JSON/msgpack/buffer equality,
actual GPU lowering, two changed-input reference evaluations, compiled file/buffer
replays, explicit HIP copies and four expected parser refusals. CPU ORT inference
and Torch HIP quadratic backward afterward pass. Selected SDK libraries and the
native `librocm_sysdeps_sqlite3.so` resolve from their required prefixes; an owned
observer confirms the SDK's isolated compiler executable.

Each prefix also passes consumer6: four GPU evaluations of broadcast/ReLU and
constant-initializer MatMul/ReLU through the public C API/C++ wrapper, plus CLI
ONNX reads for both models. Outputs match independent retained FP64 and CPU ORT
oracles; C API shapes/types and host input preservation are checked. The same
external consumer binary runs against both relocated SDKs without its old `/sdk`
mount. Existing core7/ONNXaudit4/daily5/RCCL6 bring 134 checks/eight accepted guards.
This is 68 GPU graph evaluations (60 Python, eight C API), not a model benchmark.
FP32 tolerance remains `2e-5 + 2e-5*abs(CPU_double)`, fast math false/tuning off.

No all-model/dtype/IR/opset, cross-version/device MXR, MIGraphX autograd, GPU
allocation-canary, full C API/CLI/embedded-Python, Windows or performance claim.
MIOpen fusion remains excluded for gfx1013; BF16/FP8/MX/Lt/CK/MLIR admission is
unchanged. The historical Python CPU fault is unresolved and remains distinct
from already-fixed VM-heap corruption.

## Storage, preservation and failures

Builds use the retained workstation-backed B06.4 storage, with new data in
`tmp/migraphx-b064/20261025-storage/durable/b067`. Four CPUs/jobs, link pool one,
RAM10GiB/mem+swap12GiB, read-only container root/no build GPU/network and a cloned
private2GiB ccache/no reset. Prior external16GiB admission/14GiB growth/root768MiB
floor/two-hour phase bound remain. Accepted build `logs-sdk-r5` and consumer
`consumer-r2` remain there, along with all earlier attempts/caches/source views.

Bridge disconnections interrupted preservation scanning before compilation/GPU
work. Empty interrupted wrapper files initially caused unknown-target refusals;
restoring exact B06.6 files, checking their hashes and forcing CMake configuration
recovered the copied build. No original tree was reset. Alias-aware scans check
every expected digest while reading unchanged resolved targets once; local alias,
conflicting-digest, retarget and content-drift checks pass.

Root ordinary space was initially about 773MiB, then rose to about55GiB during
this batch without a cleanup/reserve-policy action by these tools; the origin was
not established. Native qualification therefore stages the unchanged archive and
two extracts under `WS/tmp/migraphx-b067-native`, where WS is
`/home/nik/bc250work/rocm-10.2-gfx1013`. Extraction admission requires4GiB ordinary space;
its growth is checked against2GiB with a2GiB remaining floor. Stage receipt records
58,866,450,432B before and58,476,675,072B after extraction. The bridge/build trees
are no longer needed for qualification or retained native SDK use.

All original665 protected files still match:529 native paths plus136 files verified
in their preserved workstation backing before and after the native GPU batch.
The split is explicit in `external-preservation.json` and the final verification;
it does not claim reads through an absent mount. New helpers/SDK/archive/consumer
bring1,675 native protected paths,1,811 combined; all29 recorded links/four AOT DBs
match. Kernel logs match through export/cleanup on unchanged r2 boot
`e955931f-5edb-42e5-8f14-eafc2fbaf361`. Eight accepted and three failed guards
acknowledged; no owned jobs/containers/armed guards/mount/relay remain. No retained source/build/cache/image cleanup was performed; native extracts
and backing data remain. The
new source-transfer archive's original name is preserved as an alias to its
byte-verified native SDK copy.

Three failed GPU-fixture runs are retained: a stale CPU ORT postcheck path after
47 passed composites; native root-owned scratch denied JIT cache creation; and
an incorrect test expectation for SQLite's renamed SONAME after49 passed checks.
Only fixture path/ownership/loader assertions were corrected. The SDK bytes,
references and tolerances did not change; no GPU arithmetic failure was reproduced.
Transport interruption during a second pre-scan armed no GPU guard. Native jobs
use detached sessions so SSH disconnects do not terminate the batch.

## Evidence and next action

Accepted tools: `389966df4824b6e07e3accb67a679c5b2f2a5e7c`.
29 retained local stack tests, Black, Python compilation, shell syntax and C++
format checks pass. Ruff is unavailable; full upstream CI was not run.

`kernel/artifacts/migraphx-b067/evidence.tar.gz` is1,324,388B, SHA256
`27edca3ad576cb11241a27a357cd1ec56c802cd58c79799b1e80b6f57ef884ed`;
253 raw files/5,574,136B. Additional `build-evidence.tar.gz` preserves42 local
build files/1,168,331B; archive232,761B SHA256
`b2fd60ab4518e16c29088ac2249fab77076412a022a141c000f213eb6519a569`.
Native accepted logs are `WS/logs/migraphx-b067-qualification-r5`; export is
`WS/tmp/migraphx-b067-native/export-r5`. Exact commands, mounts, versions, driver
observations, source receipts and guard identities are in the evidence.

**Next B06.8 (planned):** stage/register a versioned .104 opt-in SDK with identity
checks and rollback, then qualify ordinary launcher activation against the daily
and RCCL environments. Check fresh disk headroom; preserve current defaults,
boot fallbacks and .105. ORT GPU EP and broader model/dtype/performance work remain
separate admissions.
