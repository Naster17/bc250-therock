# B03.1: native Torch local CPU Gloo readiness

This batch rebuilds the qualified patched Torch source with native distributed
support, Gloo and TensorPipe/libuv enabled. NCCL/RCCL, MPI, UCC, ibverbs and
Gloo TLS remain disabled. The existing ROCm libraries and runtime image are
retained. This adds local CPU process groups; GPU collectives, RPC and multi-board
transport require separate qualification.

CPU-only qualification passed31 checks without GPU devices exposed. Final
candidate-r2 and installed-r3 each passed **1,082 checks across18 gates**:31
new distributed checks and1,051 retained sparse/FFT/vision/audio/codec/compiled
training/profiler/core checks. All306 previous protected hashes passed before
installation; the installed baseline retains them (with the intentional active
runtime JSON update) and adds the recovery/new native identities for318 paths.
Four FFT AOT databases are preserved. Both passing whole-batch kernel logs and
all accepted gate kernel logs are unchanged. The active image remains
`506f3b262fa5fe8c29b9babec07b03078dd6f85492a9ac2cffc65955297e4349`; installed
runtime JSON SHA is `32fd45595989e09597dbb0fcfc8992595e8861c0581a49827fcf965e35a865e0`.
The integration revision is `8e6416f394f83b9ba4269eb3ec7fce63c995be98`.


The source is Torch `2b449c4abe36b896a6efb2a68e7a42d9922d5696`, not the fresh
upstream reference. Gloo is pinned at `bcd1672ee07538123ea8f4fac76832efc58fb8ef`,
TensorPipe at `2b4cd91092d335a697416b2a3cb398283246849d`, and bundled libuv at
`5152db2cbfeb5582e9c27c5ea1dba2cd9e10759b`. All tracked regular dependency files
were verified: Gloo276, TensorPipe292, libuv467. The existing host Gloo export
already contains one `__HIP_PLATFORM_HCC__` to `__HIP_PLATFORM_AMD__` guard
conversion. Its exact reconstruction patch is retained in
`build_tools/bc250/gloo-legacy-platform-b031.patch`; do not apply it twice. It is
not a newly discovered runtime fix.

## Qualified boundary

The 31-check owned fixture launches two local CPU ranks with
`python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2`.
HashStore, FileStore and the default libuv TCPStore pass set/get/add/compare-set.
Gloo checks include eight SUM dtypes (FP16/BF16/FP32/FP64/int32/int64/C64/C128),
FP64 product/min/max, broadcast, all-gather, gather, scatter, reduce,
reduce-scatter-tensor, all-to-all-single, send/receive, subgroup and monitored
barrier. FP32/FP64 functional all-reduce first gradients match analytic values.

CPU DDP checks cover FP32/FP64 models with SGD momentum and AdamW, two updates
per case. Every output, parameter and gradient is compared against an independent
non-distributed combined-batch model. Different initial rank seeds exercise rank0
parameter broadcast. Fixed FP32 rtol3e-5/atol3e-7 and FP64 rtol2e-11/atol2e-13
were retained. These are small correctness fixtures, not throughput or convergence
measurements. Checkpoint/resume, higher derivatives, GPU DDP, RPC, non-loopback
networking, multi-board operation and fault recovery remain unqualified.

## Use

For scripts using the current development prefix, native `torch.distributed`
is now available. Explicitly initialize `backend="gloo"` for the qualified CPU
path; enabling a build feature does not distribute ordinary training automatically.
For an owned CPU DDP script in a mounted project directory:

```bash
bc250-torch --env GLOO_SOCKET_IFNAME=lo -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 /path/to/owned_cpu_ddp.py
```

The script should keep its model and tensors on CPU for this qualified scope.
Use bounded thread counts when running multiple ranks. Frozen release prefixes
and unrelated Python environments retain their previous Torch builds. A single
GPU supplies no multi-GPU scaling evidence; RCCL is still absent.

## Reproduce, preservation and recovery

On `.104`, `WS=/home/nik/bc250work/rocm-10.2-gfx1013`. All host controllers run
as root through `WS/helpers/update_dev_prefix.py -- ...`, holding the existing
exclusive runtime lock. `build_distributed_b031_on_host.py` copies the old native
source/build into independent `WS/pytorch-distributed-b031`, overlays the exact
qualified source archive and runs `build_torch_b031.sh`. There are no writable
hardlinks to the old build. Shared third-party sources and ROCm are mounted read-only.
The pip-only editable build uses native gfx1013, shared20GiB ccache without
statistics reset, four jobs/four CPUs, one linker,10GiB RAM/12GiB memory+swap;
no GPU devices are exposed to the build.

`qualify_distributed_cpu_b031_on_host.py` runs CPU checks without GPU devices
(2CPUs/4GiB RAM/5GiB memory+swap). The candidate suite is
`qualify_distributed_b031_on_host.py --attempt r2`. Installation uses
`deploy_distributed_b031_on_host.py --code-revision <integration SHA>` and repeats
the full suite as installed-r3. The controller automatically restores the exact
old Torch link and runtime JSON if installation qualification fails. Explicit
rollback, under the exclusive lock:

```text
python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/update_dev_prefix.py -- python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/deploy_distributed_b031_on_host.py --rollback
```

Recovery metadata and the previous shared qualifier are in
`WS/recovery/distributed-b031-before`. Old Torch source/build, image, ROCm,
caches, checkpoints, releases and boot fallbacks remain available. The user
application is excluded (`include_training_app=False`), and `.105` is untouched.
The board returned with a new boot ID after the native build; no agent reboot
was issued and the reboot's cause was not established. The GPU preflight initially
refused a user-owned miner; CPU-only qualification exposed no GPU, and GPU testing
waited until the board was free.

The historical unexplained Python CPU segfault remains unresolved, distinct from
the fixed VM-heap/CSR and cold-hostcall queue defects and B02's known invalid-CSR
assertion cleanup trigger. This batch makes no universal correctness, stability
or performance claim.

Final identities, counts and evidence are recorded in [summary.json](summary.json).
Local `kernel/artifacts/distributed-b031/` retains the source archive, exact native
payload, raw logs, guards and recovery export. Build logs remain on `.104` in
`WS/logs/distributed-b031-build/`; final reports override the historical running
build checkpoint and stale build progress file.

Candidate-r1 stopped at the historical audio gate because its default asserted
TorchCodec was absent. The gate already supports `--codec-available`; the B03.1
suite now explicitly selects it. Numerical tolerances and check counts are
unchanged. The failed attempt and its unchanged kernel log are retained.
