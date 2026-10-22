# B03.4: broader isolated native RCCL coverage

The independent native gfx1013 candidate builds **375 specialized collective
objects** and passes **629 checks across four gates**: native550, GPU Gloo41,
CPU Gloo31 and core7. It remains outside the active prefix in
`WS/rccl-b034/install-r1`. Torch NCCL remains disabled. No production source fix,
active runtime update or speed improvement is claimed in this batch.
[Verified identities and counts](verification.json).

The source is pinned rocm-systems `38fd6ba67d0aa7491d3efaee3da98da4eb0a5727`,
already containing the two B03.3 no-tracing/direct-include fixes. No patch was
reapplied. The original B03.3 candidate, builds and caches are preserved. The
independent build copies existing build files without writable hardlinks and
extracts a fresh exact source archive. Both build and client compilation exit0.
Tools revision: `7ea133762`.

The [exact filter](kernel-filter.txt) admits Ring/Tree AllReduce Sum/Prod/MinMax,
Ring Reduce/ReduceScatter Sum, Broadcast, AllGather and SendRecv, with LL/SIMPLE
and ten input dtypes where applicable. Actual Ninja targets match the generator
preview:375 admitted/489 guarded-out specialized entries from905 generated files.
Objects comprise AllReduce270, Reduce45, ReduceScatter45, Broadcast6, AllGather6
and SendRecv3; SIMPLE201/LL174. Signed reductions alias upstream unsigned storage
kernels. Mandatory CE/DDA/one-rank code also compiles. LL128 remains excluded by
upstream gfx1013 guards; FP8/PAT/symmetric/rocSHMEM/GIN/MSCCL/ROCTX/Python and
upstream test builds are outside this selection. These boundaries are not a
claim that every excluded feature is impossible on the hardware. Compilation
of Ring/Tree/LL kernels supplies no multi-rank runtime proof.

Build limits:4 CPUs/four Ninja jobs/one linker,10GiB RAM/12GiB memory+swap;
client2 CPUs/4GiB RAM/5GiB memory+swap. Both compile containers have no GPU devices
or network. Shared20GiB ccache is retained without resetting statistics. Standard
CXX/client compilation uses ccache; RCCL's custom RCCLDEV/device-link commands
bypass the compiler launcher. Cumulative cache statistics are not batch speed
measurements. Stopped containers and sources/builds are preserved.

The C++ client uses one communicator/rank on one physical gfx1013 GPU and a
non-default HIP stream. Native550 comprises:

- 120 PreMulSum checks:ten dtypes × sizes17/257/1025 × in/out × host/device scalar.
- 420 AllReduceWithBias checks:ten dtypes × three sizes × in/out × seven reduction
  variants (Sum/Prod/Min/Max/Avg/PreMulSum host/device scalar). One-rank expected
  output is x+b or2*x+b, computed independently on CPU.
- Six ordinary identity copy/group checks and four native-device/version/loaded-
  DSO/metadata/lifecycle composite checks.

Dtypes are FP32/FP64/FP16/BF16/int8/uint8/int32/uint32/int64/uint64. Bias cases use
non-aligned offset5, prefix/suffix guards, unchanged bias and out-of-place inputs.
Integer comparisons remain integer, including uint64 near its maximum; unsigned
PreMulSum checks include modulo wraparound. Signed inputs avoid overflow.
Floating inputs use small exactly representable values, so broad rounding,
exceptional values and large numerical ranges remain unqualified. The loaded
DSO is checked against this isolated candidate. Native arithmetic executes on
GPU; single-rank identity cases do not establish transport correctness. Retained
Gloo gates include existing first-gradient/DDP comparisons but remain host-staged
for communication. RCCL autograd/Torch integration is not yet qualified.

All328 protected paths, four AOT databases and links match the baseline; four
600-second guards acknowledged. Every accepted gate and the entire batch retain
identical kernel logs. Active Torch2b449c4/image506f3b26/runtimeJSON32fd4559/native
HIPa1162308 (systems1f9609fd6), ROCm/defaults and B03.3 fallback are unchanged.
Same r2 boot2db1c7c2; no reboot/.105/BIOS/fallback changes, user application
inspection/modification/test or deletion. Historical unknown Python CPU faults
remain unresolved and distinct from fixed VM-heap/CSR/B04.2 hostcall queue defects.

Reproduce on .104 with `WS=/home/nik/bc250work/rocm-10.2-gfx1013`, invoking every
host helper through `WS/helpers/update_dev_prefix.py -- ...`:

1. `build_rccl_b034_on_host.py` builds the isolated candidate from staged source.
2. `compile_rccl_b034_on_host.py` builds the client and extends the baseline.
3. `setup_rccl_b034_on_host.py` installs the backed-up qualification helper.
4. `qualify_rccl_b034_on_host.py --attempt r1` runs the guarded629-check batch.
5. `export_rccl_b034_on_host.py` exports the accepted evidence once.

The final preservation baseline is `WS/logs/rccl-b034-qualification-r1/final.json`.
Helper backup: `WS/recovery/rccl-b034-before`. Local
`kernel/artifacts/rccl-b034` holds the source archive, native DSO/client,50 raw
logs (1,353,457 bytes), guards and recovery. The15,267,840-byte evidence archive
matches SHA256d6f1bd4944c3bec967c4729b553f59d502ae2a1a9c1911575ef0dba0ff489d4e.
Final reports supersede the historical running-build checkpoint. No own jobs or
root SSH session remain.

**Next B03.5:** inspect pinned Torch NCCL/RCCL consumers, FindNCCL and version/API
requirements. Select an independent NCCL-enabled Torch candidate against this
isolated RCCL, then bounded one-device/rank1 output and first-gradient checks
before any deployment. Keep current Gloo/runtime until qualified. No multi-GPU,
P2P/XGMI/GDA/SDMA transport, scaling or performance claim follows from one GPU.
