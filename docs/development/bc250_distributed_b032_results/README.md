# B03.2: Gloo GPU tensors and DDP on one BC-250

The installed B03.1 native Torch already handles the tested HIP tensors through
Gloo. No production patch, native rebuild, runtime replacement or default change
was needed. **108 checks across six gates pass**, including41 new GPU checks,
31 retained CPU distributed checks,28 compiled dense/CNN checks, one profiler
wrapper and seven core checks. All318 protected hashes/four FFT AOT databases
and links match; accepted gate and whole-batch kernel logs are unchanged. Six
600-second guards were acknowledged. [Identities/results](summary.json).

## What was tested

Two independent torchrun processes share `.104`'s **one** physical GPU. Both
explicitly select `cuda:0` and Gloo TCP over loopback. This is a correctness test
of the software transport and GPU autograd integration, not multi-GPU scaling.

- 24 SUM cases: FP16/BF16/FP32/FP64/int32/int64/C64/C128, sizes1/17/1025,
  offset views with untouched prefix/suffix guards. Values are small and exactly
  representable; these fixtures do not establish broad rounding accuracy.
  Async work is waited on and consumed on the submitting private HIP stream
  before the CPU reference readback.
- FP64 product/min/max/average, int64 broadcast, and FP64 all-gather/gather/
  scatter/reduce/reduce-scatter-tensor/all-to-all-single use analytic references.
- FP32/FP64 functional all-reduce outputs and first gradients match exact
  analytic values. These exercise the legacy `torch.distributed.nn.functional`
  API; its FutureWarning and the reduce-scatter alias warning are retained in
  [the raw worker log](gpu-torchrun.log). They did not fail execution. Newer
  functional collective APIs and higher derivatives remain unqualified.
- Four GPU DDP cases: FP32/FP64 with SGD momentum or AdamW, two updates each.
  Rank-specific initial seeds test rank0 parameter broadcast. Every output,
  parameter and gradient is compared to a separate CPU global-batch model of
  the same dtype. Fixed FP32 rtol3e-5/atol3e-7 and FP64 rtol2e-11/atol2e-13
  pass without relaxation. This is not distributed checkpoint or BF16 DDP
  qualification, convergence evidence or an optimizer-state identity claim.

## Where the computation runs

`ProcessGroupGlooCuda.cpp` selects `AsyncAllreduceCUDAHostWork` when the
transport reports no GPU-direct access. Gloo TCP inherits `hasGPUDirect=false`.
The work copies HIP input tensors into pinned CPU buffers, reduces there, copies
back, and uses HIP streams/events for ordering. Broadcast and other inspected
collectives have analogous host-staged paths in `ProcessGroupGloo.cpp`.
The test model's forward/backward/optimizer tensors remain on the GPU; collective
communication/reduction uses host staging. It is not RCCL or GPU-direct transport.
Exact inspected source revisions and file hashes are in [source-evidence.json](source-evidence.json).

The pinned Gloo send/recv path hands raw tensor pointers to an unbound transport
buffer, without the inspected collective staging. GPU point-to-point APIs were
not executed and remain unqualified. This batch adds no implicit CPU transfer
wrapper or global dispatch replacement. GPU coalesced/sparse collectives,
noncontiguous layouts beyond the tested offset views, non-loopback networking,
fault recovery, RPC, higher gradients and other process/device arrangements
remain separate.

## Use and reproduce

This capability is available to callers of the existing development Torch;
ordinary scripts do not become distributed automatically. An owned script must
select Gloo, initialize its process group and explicitly construct GPU DDP.
For the tested single-device arrangement, both ranks select `cuda:0` and
`DistributedDataParallel(model, device_ids=[0])`:

```bash
bc250-torch --env GLOO_SOCKET_IFNAME=lo --env GLOO_DEVICE_TRANSPORT=TCP \
  -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  /path/to/owned_gpu_ddp.py
```

Use bounded thread counts. Two processes consume additional host/GPU memory;
this test gives no reason to expect faster training on one GPU. Frozen old
prefixes retain their earlier builds.

On `.104`, run `qualify_distributed_b032_on_host.py --attempt r1` through the
existing `update_dev_prefix.py -- ...` exclusive lock. The controller refuses
other GPU clients before testing, uses the B03.1 installed318-path baseline and
excludes the user application (`include_training_app=False`). A user-owned miner
initially blocked preflight; it was left untouched and testing began after the
user freed the board. No native build or ccache operation occurred. The shared
qualifier's previous version is preserved in
`WS/recovery/distributed-b032-before/`; active runtime metadata and native DSOs
never changed. Raw evidence is `WS/logs/distributed-b032-*` and the verified local
archive/export is `kernel/artifacts/distributed-b032/`.

## RCCL disposition and next step

The pinned RCCL source in rocm-systems1f9609fd6 lists gfx1030 among its defaults,
but not gfx1013. CMake accepts explicit `GPU_TARGETS` after compiler validation;
the device header has a non-GFX9 wave32 branch. These source observations justify
a native-target feasibility build; they do not prove kernel/runtime correctness
or a hardware impossibility. RCCL remains absent and Torch's NCCL backend remains
off. No architecture spoofing, .105 changes, GPU P2P, XGMI, GDA, SDMA transport
or multi-board claims are made.

**Next B03.3:** independent bounded native gfx1013 RCCL configure/build feasibility
and single-device readiness, preserving the current Gloo/Torch prefix. A one-GPU
check cannot establish multi-GPU collective correctness or scaling. Historical
unknown Python CPU faults remain unresolved, distinct from fixed VM-heap/CSR and
B04.2 queue defects. This batch makes no universal correctness or speed claim.
