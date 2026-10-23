# B03.5: isolated Torch RCCL integration and BF16 scalar fix

The separate native gfx1013 Torch candidate enables NCCL/RCCL and passes172
checks/seven gates: Torch RCCL64, GPU Gloo41, CPU Gloo31, compiled dense14,
compiled CNN14, profiler1 and core7. It is **not installed in the active prefix**.
Daily Torch still has NCCL disabled; this batch does not change daily defaults.
[Exact identities and verification](verification.json).

Tooling revision: `ce089950f`.

The pinned ROCm path uses `cmake/External/rccl.cmake` and `find_package(rccl)`;
`FindNCCL.cmake` is the CUDA discovery path. The isolated RCCL2.32.3 headers meet
Torch's2.23 minimum. All42 header-declared API calls in the inspected consumer
files have exported symbols (including the demangled C++ diagnostic symbol).
This lexical audit is not whole-program ABI or runtime proof. The actual pip
build compiles/links the consumer and records the isolated RCCL directory in
`libtorch_hip.so`'s runtime path. No architecture spoofing or production RCCL
source change is involved. [Initial audit](consumer-audit.json).

## Defect reproduced and fixed

The unmodified Torch2b449c4 candidate passes31 initial device/backend/SUM/FP16
checks, then fails the first nonzero BF16 host-scalar PreMulSum reference: zero
instead of the expected nonzero value. `getNcclReduceOp` instantiated the BF16
helper with `float`. RCCL copies two scalar bytes for BF16, so float2.0's low
bytes produce BF16 zero. The same template also requested a float pointer from
a BF16 device tensor; that type mismatch follows directly from source inspection.
The negative GPU reproduction was the host case, not a separate device failure.

Torch commit `ade6fcc1ae7574b2f5ad285c8a2c592af1e3e99c` changes only the scalar
specialization to `at::BFloat16` and adds BF16 to the existing upstream PreMulSum
test loop. [Patch](bf16-premul.patch), [before/after hashes](bf16-fix.json).
The remote build uses the retained2b449c4 archive plus the exact production C++
overlay from ade6fcc; the updated upstream test stays in the local Torch source
and Git bundle. GPU qualification uses the owned fixture on .104.
Targeted spin static/format checks and their lintrunner `-a` invocation pass;
full upstream CI/multi-GPU tests were not run. The isolated corrected pip build
succeeds. Raw test tolerances remain unchanged.

Qualification-r2 passes all64 RCCL and41 GPU Gloo checks, then stops on an old
CPU Gloo fixture assertion requiring NCCL off. The fixture now accepts explicit
`--nccl-available`; its default still requires NCCL off. Numerical/store/worker
checks are unchanged. Qualification-r3 reruns and passes the complete172-check
batch. Neither failed run is represented as an accepted full qualification.

## GPU and backward boundary

The64 Torch/RCCL checks use one communicator/rank on **one physical gfx1013 GPU**:

- Device/backend/version and actual loaded isolated DSO/lifecycle checks:2.
- SUM with offset5/guards/private stream/async wait and immediate consumption:
  eight dtypes × sizes1/17/1025 =24. These rank-one reductions are identities.
- Nontrivial PreMulSum: FP16/BF16/FP32/FP64 × three sizes × host/device scalars =24.
  Small exactly representable quarter values and multiplier2 compare exactly
  against independent CPU arithmetic.
- Product/min/max/avg and six broadcast/reduce/gather/scatter/alltoall paths:10
  rank-one identity checks, not transport proof.
- Functional first-gradient references: FP32/FP64 =2; analytical weighted-square
  derivatives, exact comparisons for these representable inputs.
- DDP: FP32/FP64, two SGD/momentum updates with CPU output/parameter/gradient
  comparisons =2 composite checks, fixed dtype-specific tolerances.

Retained Gloo GPU checks use two local ranks sharing that GPU and host-staged
communication. RCCL uses rank1 only. No multi-GPU/Ring/Tree/LL transport,
P2P/XGMI/GDA/SDMA, scaling, speed, exceptional values, broad numerical ranges,
higher gradients, allocator registration or full-library support is claimed.
The RCCL candidate retains B03.4's filtered375 specialized objects; version flags
and exposed API names do not qualify excluded FP8/symmetric/protocol paths.

## Reproduction and preservation

On .104, `WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Run host helpers through the
existing exclusive `WS/helpers/update_dev_prefix.py -- ...` lock:

1. Stage the exact retained Torch2b449c4 archive/source metadata as
   `staging/rccl-b035/torch-source.tar` and `source.json` (the B03.1 archive is
   SHA-checked and reused). `build_rccl_b035_on_host.py` copies the active Torch
   into independent `WS/pytorch-rccl-b035`, then builds via pip only.
2. `stage_rccl_b035_on_host.py` prepares release-r1 and backs up/installs the
   shared qualification helper. `qualify_rccl_b035_on_host.py --attempt r1`
   records the BF16 negative reproduction.
3. Stage the fixed source and metadata as `helpers/b035-ProcessGroupNCCL.cpp`
   and `helpers/b035-bf16-fix.json`. `rebuild_rccl_b035_on_host.py` preserves the
   original source/native payloads and resumes the same isolated pip build.
4. Stage-r2 with `--attempt r2 --build-label rccl-b035-build-r2`, then qualify-r2;
   the preserved old CPU feature assertion is the only gate failure.
5. Back up `validate_distributed_b031.py` under `recovery/rccl-b035-before` and
   atomically install its opt-in version under the exclusive lock. Stage-r3 with
   `--attempt r3 --build-label rccl-b035-build-r2`, then qualify `--attempt r3`.
6. `export_rccl_b035_on_host.py` exports accepted and failed evidence once.

Build4 CPUs/four Ninja jobs/one linker,10GiB RAM/12GiB memory+swap; shared20GiB
ccache, no statistics reset, no network or GPU exposed during compilation.
These resource/cache records do not establish a speed improvement. The prior
B03.3/B03.4 candidates and active Torch source/build remain intact. Within the
new B03.5 build, the original source and all native library payloads are retained
in `WS/recovery/rccl-b035-bf16-before` before the fix. Historical release-r1/r2
links name the reused candidate build; use the preserved payloads for before-fix
reconstruction, not those links as immutable old binaries.

Final baseline: `WS/logs/rccl-b035-qualification-r3/final.json`,340 protected paths,
four AOT databases/links unchanged. All accepted gate and whole-batch dmesg match;
all acknowledged guards and failed-run evidence are retained. Local artifacts:
`kernel/artifacts/rccl-b035`, including native payloads and recovery. The1,080,432,640-byte archive matches
SHA256605d091e0bfbd1602c4576002425e6daed77cb78c4fd22000e9289bac81dc18b;142 raw
logs total6,797,890 bytes,11 guards acknowledged (seven accepted/four failed-run).
Active
Torch2b449c4/image506f3b26/runtimeJSON32fd4559/native HIPa1162308/systems1f9609,
ROCm/N3.2/math/SDMA0/heapoff defaults remain unchanged. Same r2 boot2db1c7c2;
no reboot/.105/BIOS/fallback/user application inspection/modification/test or
cleanup. Unknown historical Python CPU faults remain unresolved and distinct
from fixed VM-heap/CSR/B04.2 hostcall queue defects. Final verification supersedes
the historical running-build checkpoint; no own jobs/root SSH remain at handoff.

**Next B03.6:** extend isolated one-device Torch/RCCL consumer contracts (public
out-of-place/type/alias/error paths, stream/lifecycle behavior and BF16 DDP where
applicable), then broader stack regression before considering promotion. Audit
filtered/excluded API paths explicitly; keep daily runtime unchanged until that
scope is qualified. One GPU cannot establish multi-device correctness or scaling.
