# B04.5: CPU TorchCodec JPEG, PNG and WebP

**Installed:** CPU JPEG/PNG/WebP decoding and PNG/JPEG encoding through the
existing `bc250-torch` / `bc250-train` runtime. Candidate and installed runtime
each pass **203 composite checks**: image86/video55/audio46/vision9/core7.
This batch enables existing codec implementations; no new source patch or Torch
rebuild was needed. The B04.4 exact timestamp repair remains installed.

```python
from torchcodec.decoders import decode_image
from torchcodec.encoders import PngEncoder

image = decode_image("still.webp")  # CPU CHW; animated WebP yields NCHW
x = image.cuda().float().div(255).requires_grad_()
x.square().mean().backward()  # GPU tensor processing; file I/O has no autograd
PngEncoder(image.cpu()).to_file("output.png")  # supported 1/3-channel still image
```

Decode/encode are CPU operations. Transfer tensors explicitly for GPU processing.
GPU JPEG decoding is explicitly refused by this build. Existing torchvision,
audio file APIs and CPU video scope remain available. No reads, edits or tests
in the user's `finetune-n7` application occurred.

## Build and dependency

Preserved TorchCodec source remains `3eecace02076800cf6aa9be6364ed2f1f611548d`,
branch `users/Naster17/gfx1013-codec`. Integration `93a601b36` sets JPEG/PNG/WebP
ON, keeps IMAGE default OFF and CUDA/nvJPEG OFF. AVIF/GIF/HEIC remain disabled;
this is a build choice, not a hardware-impossibility claim. Existing build image
has libjpeg2.1.5/libpng1.6.43/libwebp1.3.2 development dependencies.

The old runtime lacked libwebpdemux. Its exact `libwebpdemux2`1.3.2-0.4build3
payload/copyright were copied from the existing build image into
`/opt/bc250-codec-sysdeps/{lib,licenses}`, with an image-local loader-cache entry.
No host system package/library was changed. This private copied dependency is
not registered as a runtime dpkg package. Its DSO SHA256 is
`f31bf01638c352533e18720676db66fcb190a622db5bf617558a49bc445d0b83`;
raw payload, loader configuration and notices are exported locally for recovery.
No missing dependency was hidden by disabling a codec.

Build resources: four Ninja jobs/four CPUs,8GiB memory/10GiB memory+swap, shared
20GiB ccache without reset, no GPU devices exposed. Existing ROCm prefix/native
gfx1013/Torch2.13/Python3.12/ABI1 configuration retained. Private development
wheel/distro dependencies retain the N8 release/license-packaging boundary.

## Evidence

[summary.json](summary.json) includes exact checks, mapped codec DSO hashes,
fixture identities, resources and deployment. All gates passed on their first
run with unchanged declared precision bounds.

| Image scope | Checks | Reference |
| --- | ---: | --- |
| PNG8 | 24 | Independently serialized PNG/CRC/zlib: gray, gray-alpha, RGB, RGBA; path/bytes/tensor, RGB/RGBA modes and uint16 promotion; exact samples |
| PNG16 | 6 | Independent grayscale/RGB samples; preserved native/auto precision and defined uint8 scaling |
| Orientation | 1 | Independent PNG eXIf orientation6 against explicit90-degree transform |
| JPEG | 10 | Gray/RGB path/bytes/tensor/automatic decode against separate Pillow decoder; uint16 promotion |
| WebP | 10 | Exact lossless RGB/RGBA pixels across input forms and automatic dispatch; two-frame animated pixels |
| Encoders | 24 | PNG/JPEG, gray/RGB, contiguous/strided, path/file-like/tensor; independent Pillow pixel verification |
| Refusals | 9 | Three corrupt payloads, invalid mode/dtype/quality/rank/channels; explicit unavailable GPU JPEG |
| HIP interoperability | 2 | CPU decode -> GPU spatial output/weighted input-gradient against CPU float64 -> explicit CPU PNG encode/exact independent pixels |

PNG/WebP sample comparisons are exact. JPEG cross-decoder differences allow at
most2 sample values; lossy encoded JPEG mean absolute error against the original
is bounded below20 in the owned fixtures. GPU spatial limits rtol2e-6/atol2e-7.
The malformed PNG test emits an expected libpng warning; rejection still passes.
This does not establish arbitrary malformed-input security or universal quality.

Ten600-second guards acknowledged. Every GPU gate and both whole passing batches
have identical before/after kernel logs.306 stack hashes/four FFT AOT databases/
compatibility links retained. No GPU codec, WebP encoder, AVIF/GIF/HEIC, all PNG
palette/interlace/modes, animation timing/disposal, all EXIF/HDR, speed or
universal-stability claim. Historical unknown Python CPU fault stays unresolved,
distinct from the fixed VM-heap/CSR and B04.2 queue defects.

## Installation, recovery and handoff

- Image `94c610368d0fcf596a9f9f6355a5d5d28fd1111d54d4033c46e6b5977cae1ece`.
- Wheel SHA256 `0551a76dcc4c9cc950cd8045855ae7a235c9b0f8c6edd220396828a88d19e09d`.
- Runtime JSON SHA256 `c1fd78993ec25d0866e75e69bd4cc8ef2ebe9710f9a1f08e0de33bb350a2aaea`.

Only a new codec wheel/private image dependency/image and dev runtime metadata
changed. Existing native ROCm/Torch/kpacks/N3.2/math/SDMA0/heap-off defaults remain.
Prior image `87875c40998bf083ecbc991e7dbe00b4e5549755b84d0f5b36d470167ab5de6d`,
source/build/cache/release/checkpoint history retained. Recovery is
`WS/recovery/image-b045-before/runtime.json`, including original shared helper,
where `WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Run as root under exclusive lock:

```sh
python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/update_dev_prefix.py -- python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/deploy_image_b045_on_host.py --rollback
```

Rollback unused; failed installed gates would restore prior metadata automatically.
No image or source deletion is part of recovery. Active baseline:
`WS/logs/image-b045-qualification-r3/final.json`.

Local `kernel/artifacts/image-b045` retains source/provenance/wheel/runtime
sysdeps/notices/loader config/recovery/build manifests and218 SHA-verified raw
log files2,870,579 bytes, plus ten guard states. Raw remote `WS/logs/image-b045-*`.
Export archive SHA256:
`97879a5e175576bd8ba8cececf6e29267ed192d7fc03d7b11f9b58820e4947a2`.
Same r2 bootc646d874; no agent reboot/.105/BIOS/fallback/delete change. Disk
21,467,881,472 bytes free observed. No own running build/test/container/guard/export;
root SSH sessions closed.

**Next B01.1:** review N2.14/B01 BF16 FFT evidence and current autocast/promotion
paths before coding. Qualify any missing explicit FP32 promotion helper's
complex64 outputs/input gradients. Common media scope above is bounded; optional
codec expansion remains separate rather than an exhaustive combination sweep.
