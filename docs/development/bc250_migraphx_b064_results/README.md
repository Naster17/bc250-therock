# B06.4: isolated native FP32 MIGraphX graphs

The isolated minimal GPU backend is built and its owned FP32 graph gate passes
on native gfx1013 wave32. Both full 44-check passes and final preservation/export checks pass. No active
ROCm/Torch prefix or runtime image was changed, and no ONNX GPU provider was
installed. Source `622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83` adds the
rocBLAS-only query repair over B06.3 `35b40656d`. Tools: TheRock `f8635babe`.

## Build storage and isolation

The .104 root filesystem had approximately 1 GiB of ordinary available space.
Its reserved ext4 blocks were excluded from build admission. Dedicated backing
storage is the workstation directory
`tmp/migraphx-b064/20261025-storage/durable` (initially about 109 GiB available),
served through an SSHFS slave over the existing encrypted SSH connection.
This introduces no listening service, SSH key, SSH configuration change or boot
mount. Artifacts remain on workstation disk if the transport stops. A failed
transport can interrupt the candidate build; no active runtime depends on it.

On .104 the temporary mount is
`WS/tmp/migraphx-b064/20261025-storage/mount`. SSHFS 3.7.6-r0 and FUSE 3.18.3-r0
were installed and the existing r2 FUSE module loaded. The inherited APK
repository list also attempted its .105 mirror; that index request was refused.
No .105 deployment, application action or state change was performed.

Storage passed file fsync, mapped writes/flush, atomic rename, relative symlink
and local-client locking checks before building. Source is extracted from the
exact committed archive into `WS/tmp/migraphx-b064/20261025-source` and mounted
read-only inside the build container; this small native copy avoids repeated
network header reads. Build outputs and the independent 2 GiB ccache stay on
external storage; the old shared cache is untouched. Initial network source
reads were slow, so the owned build was stopped and resumed with native inputs,
retaining completed objects and both original logs. No existing artifact was
deleted or filesystem reserve policy changed.

The controller admits at least 16 GiB of ordinary external space and stops its
own container if external available space drops by 14 GiB, .104 ordinary space
falls below 768 MiB, or the attempt reaches two hours. Four CPUs/four compile
jobs, one link slot, RAM 10 GiB and memory+swap 12 GiB; read-only container root,
no build GPU/network access. Release host code uses O2/g0, not a performance
qualification. The minimal profile keeps HIPRTC/rocBLAS/MIOpen on and
Lt/CK/rocMLIR/AMDMLSS/Eigen/ONNX/TF/Python off.

## Replaying the candidate

Use the committed B06.4 tools under `build_tools/bc250/`. The exact commands,
container mounts and exits are retained in each `progress.json`/`final.json`.
The wrapper `migraphx_b064_CMakeLists.txt` adds the owned fixture to the source;
its root `test`, `src`, `cmake` and `LICENSE` links point into that source view.
These accommodate upstream source-root assumptions without changing MIGraphX.
The initial wrapper omitted these views and refused during configuration;
that failure/cache/log are retained.

Prepare the temporary mount and its mode-0600 `input`/`output` FIFOs on .104.
Start `sshfs -f -o slave -o allow_other :BACKING_DIRECTORY MOUNT` with its stdin
from `input` and stdout to `output`, then run locally:

```sh
python3 therock/build_tools/bc250/bridge_migraphx_b064.py \
  --log-directory tmp/migraphx-b064/20261025-storage
```

The bridge connects the local OpenSSH SFTP server to those FIFOs using SSH.
Do not treat this temporary mount as a production dependency. Keep backing
artifacts, caches and old stopped containers when unmounting it.

In the root development shell on .104, helpers run in this order:

```sh
python3 WS/helpers/build_migraphx_b064_on_host.py --attempt r4
python3 WS/helpers/finish_migraphx_b064_on_host.py
python3 WS/helpers/qualify_migraphx_b064_on_host.py
python3 WS/helpers/export_migraphx_b064_on_host.py
```

`WS` denotes `/home/nik/bc250work/rocm-10.2-gfx1013`, not a literal shell variable.
Attempt names and output directories are exclusive; choose new ones when
replaying rather than overwriting retained evidence. The build tree supports
incremental resume. The source archive, wrapper, fixture, flags and dependency
paths must match their recorded identities.

## Executed reference scope

The owned fixture compiled hand-built FP32 graphs and checked actual GPU-lowered
operations. It covers strided/broadcast pointwise/ReLU, row sums, rocBLAS GEMM
and MIOpen convolution, with two changed-input evaluations per compiled graph.
Independent CPU-double references check outputs; host input buffers, padding and
guards must remain unchanged. This does not qualify GPU allocation canaries,
all layouts/dtypes, graph autograd, models or speed. The frontend/provider and
low-precision limitations from B06.3 remain until separately tested.

The fixture passed strict C++17 syntax checking with the generated candidate
headers. Its initial missing `instruction.hpp` include was corrected; original
syntax diagnostics are preserved. The actual GPU gate passes 22 composites: native device identity, both
rocBLAS-only query overloads, and 20 graph evaluations across 10 compiled graphs.
Pointwise widths 17/33 use row stride width+5 and broadcast bias; reductions
cover widths 17/33/129/513; GEMM is 3×K by K×5 with K=7/17; convolution is
N×2×5×5 with 3×2×3×3 weights, N=1/2. Compiled IR contains `gpu::code_object`,
`gpu::gemm` and `gpu::convolution`, with explicit HIP input/output copies.
`fast_math=false`, exhaustive tuning off, float32 data only. Comparisons use
`abs(error) <= 2e-5 + 2e-5*abs(CPU_double_reference)`; no test was relaxed.
The loader record resolves candidate MIGraphX and the existing selected native
HIP/HIPRTC/HSA/rocBLAS/MIOpen/SQLite DSOs. Existing hipBLASLt may load transitively;
MIGraphX's own hipBLASLt dispatch is disabled and both support queries return false.
Five extracted device objects identify gfx1013 and their disassembly contains no
MFMA/WMMA instructions; that statement is scoped to these objects.

Retained core 7/ONNX audit 4/daily consumers 5/RCCL consumers 6 make 44 checks
per qualification pass. Both numerical passes completed. The final pass freezes
tools after export-only audit additions during the first pass's postcheck;
original records remain intact. The test source/graph arithmetic did not change.
Final SHA/link/kernel-log/guard verification and archive details are recorded in
`kernel/artifacts/migraphx-b064/verification.json` and the local verification.
No universal operation/layout/dtype, graph autograd, model speed or stability claim.
Historical Python CPU segfault remains unresolved and distinct from fixed VM-heap
corruption.


## Optional-backend repair and next action

The first full GPU build reproduced `lowering.cpp:288` referring to
`gfx_default_rocblas` while hipBLASLt was disabled. Both declarations and
implementations had been hidden behind the optional-backend guard. They are now
available in both configurations; without hipBLASLt they return true without
querying a device. The enabled-backend implementation is preserved. A regression
is added in `test/gpu/device_name.cpp`, both compile configurations pass syntax
checks, and the actual linked no-Lt overloads pass on .104. Neither the hardware
whitelist nor the BF16/FP8/MX predicates were broadened. The disabled rocMLIR
host code still emits unused-helper warnings; no warning suppression was added.

**Next B06.5:** audit and build pinned native ONNX parser prerequisites on the
same durable storage, then qualify small ONNX→gfx1013 FP32 graph cases against
independent CPU references. Keep the current CPU ONNX runtime and Torch defaults;
MIGraphX parser/bindings/ORT provider, low precision and larger model coverage are
separate admissions. The candidate build and cache remain preserved; reconnect
the temporary transport before using their .104 mount paths.


The accepted source has 4,782 archived files; all match the native .104 input
snapshot. All 172 pinned rocm-cmake files match their preserved archive.
537 protected hashes/four AOT DBs/links remain unchanged; all ten guards in the
two reports acknowledged, and kernel logs remain identical through final unmount.
Same r2 boot `c6a50cfa-a368-4ba5-ab28-66250b4f1f1f`; no runtime image/prefix,
boot/fallback/BIOS or user application change. 29 local stack tests, Black,
Python compilation, shell syntax and changed C++ formatting pass; Ruff was
unavailable and full upstream CI was not run. The source repair bundle was
reconstructed on the preserved upstream/B06.3 chain and passed Git fsck.

Accepted archive: 67,753,287 bytes,
SHA256 `b3c1f4576ec1e9bef91458def557080d340e99ea4cfbf7742d72820f48230c32`.
154 extracted raw log/guard/diagnostic files total 14,184,780 bytes.
Build tree approximately 426 MiB, cache approximately 43 MiB. Export measured
996,155,392 bytes ordinary .104 root headroom and 116,659,523,584 bytes on the
backing filesystem before writing the archive. The temporary mount and relay are
closed; no owned job/container/armed guard remains. Backing build/cache/source
files and stopped build containers remain preserved. After unmount, terminate
the owned bridge process with SIGTERM to close the FIFO relay's SSH/SFTP children;
its server exit from that deliberate shutdown is unrelated to GPU test status.
