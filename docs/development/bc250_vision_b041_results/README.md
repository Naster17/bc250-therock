# B04.1: native gfx1013 torchvision

Completed 2026-10-20. Preparation `626e19ef2`, integration `220bd80e5`.
Torchvision **0.28.0+gfx1013** and Pillow **13.0.0** are installed in the
active development container image used by `bc250-torch` / `bc250-train`.
Host Python environments outside those launchers do not acquire these packages.

Torchvision source is preserved locally at `therock/sources/bc250-torchvision`,
branch `release/0.28`, revision `8fb87713a24951e639c494b0f2a8a81b5f8e33a6`
(tag v0.28.0). The pinned upstream CI targets release/2.13. It was built against
our existing patched Torch **2.13.0a0+gfx1013**, HIP7.17.0, Python3.12 and C++11 ABI1.
No torchvision operator source patch or replacement Torch wheel was needed.

## Qualified boundaries

Each passing candidate and installed batch has **67 composite checks**:

| Path | Checks | Reference and boundary |
| --- | ---: | --- |
| NMS / batched NMS | 12 | Independent greedy CPU FP64; empty,5,73 boxes; distinct scores/categories; FP16/FP32/FP64 |
| ROI Align / ROI Pool / PS ROI Align / PS ROI Pool | 28 | CPU FP64 output/input gradients; FP16/FP32/FP64; contiguous/strided inputs; four empty forward/zero-gradient cases |
| ROI oracle | 3 | Independent bilinear CPU FP64 output/weighted input-gradient checks; aligned/unaligned and fixed/adaptive sampling |
| Deformable convolution | 7 | Six CPU FP64 output/all-gradient cases, grouped weights and offset groups2, masks on/off, FP16/FP32/FP64; separate CPU zero-offset identity versus ordinary grouped convolution |
| GPU trace replay | 1 | CPU-reference output/backward; actual native ROI forward/backward GPU kernels recorded |
| CPU codecs + HIP transfer | 3 | PNG exact roundtrip, JPEG against Pillow atol3, lossless WebP against Pillow; transfer preserves bytes |
| Tensor transforms | 4 | Resize with antialias, normalization, affine, crop: FP32 GPU output/weighted input gradient against CPU FP64 |
| Synthetic ResNet18 | 1 | Random weights, batch2/32x32, eval-mode FP32; CPU FP32 output/input/all62 parameter gradients; no downloads |
| Package identity | 1 | Existing Torch/HIP and new vision/Pillow import |
| Retained core | 7 | GEMM FP32/BF16, math SDPA D32/D256, LayerNorm, FFT forward/backward and exact RNG replay; consumer imports |

Maximum observed relativeL2: operators0.001519, transforms5.986e-7. Operator
limits remain FP64 rtol1e-9/atol1e-10/L2<1e-8; FP32 .002/2e-5/.003;
FP16 .05/.008/.04. No tolerance was relaxed. ResNet uses the FP32 limits.
NMS has no promised derivative; ROI coordinates are metadata, not differentiated.
Deformable convolution checks input/offset/weight/bias and optional mask gradients.
No throughput, universal model/shape/layout, training-mode ResNet, second derivative,
BF16 custom vision operator, GPU decoding, video or distributed claim.
The pinned native custom kernels dispatch FP16/FP32/FP64; BF16 is not admitted.
RPP/rocJPEG/rocDecode are separate components, not automatically selected here.

## Build, packaging and failures retained

`WS` means `/home/nik/bc250work/rocm-10.2-gfx1013` on .104. All development
commands ran under `WS/helpers/update_dev_prefix.py -- COMMAND...`.

- Isolated build source `WS/vision-b041/source` preserves the copied Git checkout;
  its source/build outputs and raw wheel remain. FORCE_CUDA1 selects HIP despite
  no GPU devices exposed to the build container; target is native gfx1013 only.
- Four Ninja jobs,10GiB build memory/12GiB memory+swap limit, eight CPU allocation;
  shared retained20GiB ccache. HIP compiler explicitly runs through ccache.
  Torch's HIP cpp-extension wrapper bypasses automatic CPU compiler wrapping;
  initial CPU translation units are uncached. Aggregate cache totals are not a
  batch-specific hit-rate measurement.
- Separate build image installs JPEG/PNG/WebP development packages and patchelf;
  retained build/runtime images are unchanged. See build-r3 logs for exact versions.
- Repaired wheel bundles libjpeg/libpng/libwebp/libsharpyuv with license notices;
  each codec has its own `$ORIGIN` RUNPATH and image.so uses `$ORIGIN/codec-libs`.
  Wheel RECORD is regenerated. Both wheel variants are retained.
- First preflight refused two changed user-owned training documents; all other311
  identities matched. Both documents were preserved, entry drift recorded in
  `vision-b041-entry-r2/final.json`; no trainer restoration/requalification claim.
- Build-r2 failed because host `/var/tmp` is absent. Build-r3 uses an explicit
  workspace TMPDIR; no source/build/cache reset was necessary. Package-r1 lacked
  an output directory; package-r2 created it, preserving failed staging.
- Candidate-r1 passed51 operator checks but failed codec loading: image.so RUNPATH
  did not cover WebP's indirect libsharpyuv dependency. CPU-only loader diagnosis
  proved the missing DSO. Package-r3 adds each bundled codec's relative RUNPATH;
  candidate-r2 and installed-r3 pass all67 checks each. Failed image/wheels/logs
  remain; none of these failures were hidden or promoted to passing evidence.

## Runtime, use and rollback

New immutable runtime image:
`a0b85fa88c9ecfbec88a8774f19db0fbc18fcfeaa086a4e402b69aa99ba55e27`.
Previous independent image:
`5df0a58392de13916819f6d84342b0338ae2c2bbb2f3c21a1f3ad38be2724a59`.

Only the development runtime.json image selection and new torchvision metadata
changed. Original JSON is `WS/recovery/vision-b041-before/runtime.json`.
The old release and its consumer site-packages are untouched; the new packages
live in `/opt/torch-venv` inside the new image. Qualified Torch source/native
libraries, ROCm libraries, N3.2 CSV, math SDPA, SDMA0/heap-off and boot fallbacks
remain. Shared qualification helper gained an optional candidate-release argument;
its previous copy is in the same recovery directory.

Example through the existing launcher:

```sh
bc250-torch -c 'import torch, torchvision; b=torch.tensor([[0.,0.,2.,2.],[0.,0.,1.,1.]],device="cuda"); s=torch.tensor([.9,.8],device="cuda"); print(torchvision.__version__,torchvision.ops.nms(b,s,.3))'
```

CPU image decoding is explicit: decode on CPU, then transfer tensors with
`.to("cuda")` for supported transforms/model computation. This installation adds
availability; it does not silently rewrite applications or accelerate LoRA.

Exact rollback, run as root on .104:

```sh
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/deploy_vision_b041_on_host.py --rollback
```

Replace `WS` with the absolute path above. The helper refuses an unexpected
current runtime identity, then atomically restores the byte-exact old JSON.
It retains all images, packages, builds, caches and evidence.

Eight600s guards acknowledged, including the failed candidate codec gate. Kernel
logs match within every GPU gate and each qualification batch. Entry/current
r2 boot48bdf727-b2f1-4d7b-aba5-2a0b775f1387 unchanged; no reboot/.105/BIOS change.
All313 protected identities match final installed baseline; only development
runtime.json intentionally differs from entry. Four FFT AOT databases and links
are retained. Disk18,191,425,536B free observed; no cleanup this batch.
No agent build/test/container/guard deadline remains; root SSH closed.
Historical Python CPU segfault remains unresolved, separate from fixed VM-heap/CSR
corruption and the historical N3.8 host OOM.

Raw reports: `WS/logs/vision-b041-*`, active preservation baseline
`WS/logs/vision-b041-qualification-r3/final.json`. Durable local artifact root:
`kernel/artifacts/vision-b041`, with SHA-verified source archive, raw/repaired
wheels, Pillow wheel and all109 raw log files in `remote/logs`.
[Summary](summary.json) records pins, hashes and scope.
Next **B04.2**: inspect pinned Torch-compatible torchaudio build/API/backend
requirements, select hardware-feasible audio tensor operations and independent
CPU references, then isolated candidate build/test before installation.
