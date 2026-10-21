# B04.4: CPU video reference qualification and exact timestamp repair

The installed TorchCodec0.17.0+gfx1013 now has bounded CPU video decode, seek and
encode qualification, plus a repaired exact single-frame timestamp API. Candidate
and installed runtime each pass **108 composite checks**: video55/audio46/core7.
No changes, reads or tests in `finetune-n7`; all fixtures belong to the stack.

Use through `bc250-torch` or `bc250-train`:

```python
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder

clip = VideoDecoder("input.mkv", device="cpu")
frames = clip.get_frames_at([0, 2, 4]).data  # CPU uint8 NCHW
x = frames.cuda().float().div(255).requires_grad_()
# GPU tensor operations can participate in autograd. File I/O does not.
y = x.mean(dim=(-2, -1))
y.square().mean().backward()
VideoEncoder(frames, frame_rate=5).to_file(
    "output.mkv", codec="ffv1", pixel_format="bgr0"
)
```

Decode/encode run on CPU. GPU media acceleration remains off. Existing
Torch/Torchvision/Torchaudio and CPU audio file APIs remain available; no host
Python environments were rewritten.

## Demonstrated defect and repair

Original runtime completed40 checks, then failed the variable-timestamp scalar
seek. The owned nine-frame clip has independently prescribed timestamps
`t[i]=i*i/5`. Packet durations are shorter than some adjacent timestamp gaps.
The single-frame API used packet duration and accepted the next frame in a gap;
index access and the batch timestamp API use the scanned next presentation
timestamp. This contradicted the pinned player's stated next-PTS behavior.

Source `3eecace02076800cf6aa9be6364ed2f1f611548d` changes
`SingleStreamDecoder::get_frame_played_at` to use the scanned index in exact and
custom-mapping modes. The approximate branch is unchanged. Returned packet
metadata durations are not rewritten. Original fixture/oracles/tolerances remain
unchanged; the repaired candidate and installed runtime pass them. This batch
qualifies the default exact mode; custom mappings and approximate VFR seeking
remain separate coverage. Original failure/logs are retained.

Preserved Git: `sources/bc250-torchcodec`, branch `users/Naster17/gfx1013-codec`,
base `d9dc54238e173000bc3969eb3959777026ff9ae8`; reviewable patch
`patches/amd-mainline/torchcodec/0001-fix-exact-single-frame-timestamp-seeks.patch`.
Integration controllers `22f796252`. The patch concerns CPU timestamp semantics;
it is distinct from the historical unknown Python CPU execution fault, fixed
VM-heap/CSR corruption and B04.2's HIP cold-hostcall queue repair.

## Test scope

[Machine-readable results](summary.json) contain pins, checks, fixture hashes,
actual encoded picture types and deployment evidence.

| Scope | Checks | Reference |
| --- | ---: | --- |
| Constant-rate decoding/seeking | 32 | FFV1 and lossless H264-RGB,9 frames26x18; NCHW/NHWC, exact original RGB bytes, PTS/duration, backwards/forwards/negative-index access, duplicate/unsorted batches, slices/ranges and time seeks |
| Memory sources | 6 | Both codecs via bytes, file-like object and uint8 encoded tensor |
| Variable timestamps | 3 | Original pixel bytes and prescribed PTS, index seeks and time seeks across packet-duration gaps |
| Encoding | 7 | Both codecs, contiguous/strided input, independent FFmpeg CLI pixel comparison, encoded tensor/file-like output and two incremental chunks |
| Refusals | 5 | Missing file, invalid dimension order/index/dtype/frame rate |
| Explicit HIP interoperability | 2 | Decoded CPU frames -> GPU average-pooling output/input gradient against CPU float64 -> explicit CPU transfer/independent lossless encode verification |
| Retained audio/core | 53 | B04.3 file codec46 and core7 forward/backward/RNG/consumer checks |

Pixel comparisons are exact. PTS/duration tolerance1e-9; GPU spatial output and
weighted input-gradient limits rtol2e-6/atol2e-7. The owned H264-RGB clip contains
I/P frames; this does not qualify B-frame behavior, YUV conversion, HDR, arbitrary
containers, all models/formats, long streaming, GPU codec acceleration, speed or
universal stability. CUDA/image codec builds remain off; existing torchvision
image support remains available.

Seven600-second guards acknowledged, including the original failed numerical
gate. Guard acknowledgment records SSH return, not numerical success. Both
passing whole batches and their individual GPU gates have identical before/after
kernel logs.306 protected stack paths/four FFT AOT databases/compatibility links
remain; user application paths stay excluded, `include_training_app=False`.

## Installation, recovery and retained artifacts

Only the TorchCodec wheel in a new image and development runtime image/metadata
changed. Existing native ROCm/Torch DSOs/kpacks/N3.2/math SDPA/SDMA0/heap-off
settings and B04.2 CLR repair remain. Wheel version stays0.17.0+gfx1013; source
revision and SHA distinguish this repair from B04.3.

- Image: `87875c40998bf083ecbc991e7dbe00b4e5549755b84d0f5b36d470167ab5de6d`.
- Wheel SHA256: `72eee43277a2806684efa1f485f951567e935582a02bb731ced6f24ca65ef8f3`.
- Runtime JSON SHA256: `4cf699ed5c85b6eccf76b04976ca6abc9c83bd0449474ff8f28b2031a6a5cb92`.
- Previous image `dcd025fbee1ffd180b9ff4f3ab8b753f2415d8a58914d77506ea12d4777a88d7` remains.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Recovery is
`WS/recovery/video-b044-before/runtime.json`; original shared helper also retained.
Run as root under the exclusive prefix lock for exact old runtime metadata/image:

```sh
python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/update_dev_prefix.py -- python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/deploy_video_b044_on_host.py --rollback
```

Rollback was not used. Failed installed tests would automatically restore the
previous metadata. Source/build/image/cache history is retained. Build4 Ninja/
4 CPUs/8GiB memory/10GiB memory+swap/shared20GiB ccache; no build GPU devices.
Existing FFmpeg build image reused, no new distro package installation. Private
FFmpeg development packaging retains the B04.3 N8 release/license boundary.

Raw `WS/logs/video-b044-*` is durably copied/SHA-verified locally in
`kernel/artifacts/video-b044/remote`:130 raw log files2,536,706 bytes, seven guard
states, wheel, recovery and build package manifests. Source archive/provenance,
Git source and patch are preserved locally. Export archive SHA256:
`8295c64f69c482a1bc4de090cc6d3bdef3539673c61b10b6eb763f18fc4934f3`.
Current baseline: `WS/logs/video-b044-qualification-r3/final.json`.

Boot `c646d874-ebd6-4e4b-bb77-2e47233748e0` was already new at entry, still
7.1.5-bc250rocm-r2; same throughout tests. No agent reboot/.105/BIOS/fallback or
deletion change. Disk21,589,233,664 bytes free observed; no cleanup. No own running
build/test/container/guard/export; root SSH sessions closed.

**Next B04.5:** inspect pinned CPU image codec flags/dependencies, enable feasible
TorchCodec JPEG/PNG/WebP paths and qualify owned independent pixel references.
Keep GPU codec paths off. Broader B01/B02/B03/B05/B06/B07/N2–N8 remain open.
