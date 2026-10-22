# B03.3: isolated native RCCL build and single-device readiness

A restricted RCCL candidate now builds for **native gfx1013** and passes82
single-device checks. Together with retained GPU Gloo41/CPU Gloo31/core7, the
batch passes **161 checks across four gates**. The candidate remains in
`WS/rccl-b033/install-r1`, outside the active prefix. Torch's NCCL backend remains
off. No runtime metadata, image, Torch/ROCm DSO, kernel or default changed.
[Exact identities and results](summary.json).

## Build boundary and fixes

The source archive is pinned rocm-systems1f9609fd6 plus two small source fixes:
`c653cbc33c` makes direct `nvtx.h` includes honor `NVTX_NO_IMPL` and adds the
missing no-op NCCL range macro; `38fd6ba67d` adds the direct `<iostream>` include
used by DDA IPC diagnostics. These fix host compilation with tracing disabled;
no collective arithmetic kernel was patched. Original license headers and the
IPC source's CRLF line endings are retained. The source patch is
`build_tools/bc250/rccl-no-tracing-b033.patch`, applied from the rocm-systems root.

Build-r1 completed native device linking but failed on duplicate `nccl_domain`
definitions from the full/stub NVTX headers. Build-r2 passed that point, then
failed because IPC's `std::cerr` relied on the old transitive tracing include.
Build-r3 and the reference client build succeeded. All failed logs, original
source files and the initial source archive are retained; the same independent
build directory was resumed without resetting it or deleting its cache.

CMake explicitly selects `GPU_TARGETS=gfx1013`; no architecture spoofing is used.
`ONLY_FUNCS="AllReduce RING SIMPLE Sum f32"` restricts the generated collective
kernel family. Other mandatory device code, including one-rank kernels and
CE/DDA units, also compiles. Symmetric kernel generation, rocSHMEM/GIN, MSCCL,
ROCTX, Python bindings and upstream test builds are off. This is a feasibility
candidate, **not a full RCCL distribution**. Compilation of the Ring kernel does
not establish its multi-rank runtime correctness.

The build uses the existing build image/toolchain,4 CPUs/four Ninja jobs/one
linker,10GiB RAM/12GiB memory+swap, shared20GiB ccache without resetting statistics,
no network or GPU devices. The client build uses2 CPUs/4GiB RAM/5GiB memory+swap
and no GPU. Cumulative ccache statistics are not a batch timing result.

## What actually ran on the GPU

The owned C++ client verifies gfx1013, header/library versions, the loaded
candidate DSO path, one-rank communicator metadata, asynchronous status and
finalize/destroy. **72 nontrivial PreMulSum checks** cover six dtypes
(FP32/FP64/FP16/BF16/int32/int64), sizes17/257/1025, in-place/out-of-place and
host-immediate/device scalars. Input/output guards and unchanged out-of-place
inputs are checked exactly against independent CPU values. A non-default HIP
stream orders input copies, the RCCL operation and readback. Small exactly
representable inputs multiplied by2 do not qualify exceptional values, overflow
or broad floating-point accuracy.

The inspected `src/device/onerank.cu` explicitly launches `oneRankReduce` for
PreMulSum, so these cases exercise device arithmetic. Six ordinary one-rank
collective/group checks are identity transfers; they are reported separately
and supply **no multi-GPU communication proof**. No autograd integration,
scaling, performance, multi-board/P2P/XGMI/GDA/SDMA transport or full-protocol
claim is made. Networking is restricted to loopback, IB/P2P and cuMem paths are
disabled for this probe; current HSA/HIP safe defaults are preserved.

## Evidence, reproduction and next step

On `.104`, `WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Run each host helper through
`WS/helpers/update_dev_prefix.py -- ...` using the existing exclusive lock:

1. `build_rccl_b033_on_host.py` creates the independent source/build; failed
   attempts remain in `logs/rccl-b033-build-r1` and `-r2`.
2. The recorded staged overlay contains exact before/after source hashes.
   `build_rccl_b033_on_host.py --attempt r3` resumes the preserved build.
3. `compile_rccl_b033_on_host.py --build-attempt r3` builds the native reference
   client and creates the candidate preservation baseline.
4. `qualify_rccl_b033_on_host.py --attempt r1` runs the four guarded gates.

The final baseline is `WS/logs/rccl-b033-qualification-r1/final.json`:323 paths
(the prior318 plus isolated candidate identities), four FFT AOT DBs and links
match. Four600-second guards acknowledged; all gate and whole-batch kernel logs
match. Raw logs/native library/client/source archive/recovery are SHA-verified
locally under `kernel/artifacts/rccl-b033/`. Source originals are in
`WS/recovery/rccl-b033-source-before`; the shared helper is backed up separately
in `WS/recovery/rccl-b033-before`. The active prefix needs no rollback because it
was never changed. Keep the filtered candidate outside ordinary Torch use.

The board remains on the same r2 boot; no reboot, .105/BIOS/fallback changes,
user application inspection/test/modification or cleanup occurred. Historical
unknown Python CPU faults remain unresolved, distinct from fixed VM-heap/CSR
and B04.2 hostcall queue defects. The new fixes address compilation only.

**Next B03.4:** broaden the isolated gfx1013 RCCL build beyond the FP32 Ring/SIMPLE
filter, inspect admitted type/protocol combinations and qualify feasible
single-device operations before considering Torch NCCL integration. One GPU
cannot establish multi-GPU collective correctness or scaling.
