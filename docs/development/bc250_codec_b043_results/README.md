# B04.3: CPU audio file I/O through TorchCodec

**Installed on .104:** TorchCodec0.17.0+gfx1013 and FFmpeg6.1.1 in the independent
development runtime image. `torchaudio.load` and `torchaudio.save` now work through
`bc250-torch` and `bc250-train`. Existing patched Torch, native audio GPU operators,
ROCm libraries, safe defaults and previous image are retained. Host Python
virtual environments were not rewritten. No changes or tests were made in the
user's `finetune-n7` application.

Audio codecs run on the CPU. Transfer a decoded tensor explicitly to `cuda` for
HIP signal operations; transfer results back to CPU before saving. For example:

```python
import torch
import torchaudio

samples, rate = torchaudio.load("input.wav")
x = samples.cuda().requires_grad_()
y = torchaudio.functional.resample(x, rate, 8000)
y.square().mean().backward()
torch.cuda.synchronize()
torchaudio.save("output.wav", y.detach().cpu(), 8000)
```

Run such a script with `bc250-torch script.py` in the development runtime. The
file boundary has no autograd history; the GPU tensor operations have their
ordinary backward behavior. The existing torchaudio wrapper always decodes to
normalized float32. `normalize=False` warns and still returns float32; some
legacy backend/encoding options are also ignored with warnings. This batch does
not change those upstream API contracts.

## Pins and implementation

| Item | Identity |
| --- | --- |
| Preserved source | `sources/bc250-torchcodec`, branch `users/Naster17/gfx1013-codec` |
| Upstream source | v0.17.0, `d9dc54238e173000bc3969eb3959777026ff9ae8` |
| Source patches | None; source archive matches all471 tracked Git files |
| Integration code | `9c5199ccb` |
| Installed wheel SHA256 | `5850732408eaf7ef372d570be4a8367948e69181e9b4eb58f6d833462f2c7780` |
| Installed image | `dcd025fbee1ffd180b9ff4f3ab8b753f2415d8a58914d77506ea12d4777a88d7` |
| Runtime JSON SHA256 | `f902e0bc3f7c4275db9ea7f7a2234f29fcc6fd9d78b449bf95666b00ed8ccd96` |
| Retained previous image | `97fac19f0d5747265c4ff8328f99d2eaacaf8ff62e5092d22cda300f7130a632` |

The pinned [upstream CMake](https://github.com/meta-pytorch/torchcodec/blob/d9dc54238e173000bc3969eb3959777026ff9ae8/src/torchcodec/_core/CMakeLists.txt)
uses the Torch stable API target2.11. The actual build/test uses our qualified
Torch2.13.0a0+gfx1013/Python3.12/CXX11 ABI1. `ENABLE_CUDA=OFF` and
`TORCHCODEC_BUILD_IMAGE=OFF`; image targets contain explicit unsupported stubs.
This is a CPU FFmpeg6 audio build, with real native HIP interoperability checked
separately. The Torch CMake imports still require the qualified ROCm prefix and
`PYTORCH_ROCM_ARCH=gfx1013`; no architecture spoofing or replacement Torch wheel.

One isolated build image and one candidate runtime image were added. Builds
use four Ninja jobs/four CPUs,8GiB memory/10GiB memory+swap and the shared20GiB
ccache, explicitly configured as the C/C++ launcher. No GPU devices are exposed
to the build container. Source/build/cache/image history remains. The local
private-development wheel links distro FFmpeg; copyright/license files are
retained in the images. Distributable release dependency/license packaging is
still an N8 task.

## Qualification

**136 candidate and136 installed composite checks pass**, with unchanged
precision gates. See [summary.json](summary.json) for detailed checks, pinned
libraries, numerical results and recovery identities.

| Gate | Checks per runtime | Evidence |
| --- | ---: | --- |
| CPU file codecs + HIP interop | 46 | Independent PCM16 mono/stereo and PCM24 stereo RIFF fixtures; FLAC; path/file-like/channel layout/crop/empty reads; direct decoder metadata; MP3 versus separate FFmpeg CLI; WAV/FLAC saves including strided/time-major/1D input; independently parsed WAV samples; direct encoded bytes; warnings/refusals; GPU resample output/input gradient and explicit CPU save |
| Retained audio DSP | 34 | FP32/64 output/input gradients, strided layouts, CPU double and explicit DFT references |
| Retained native audio | 40 | IIR waveform/coefficient gradients and GPU stream trace; RNNT costs/logit gradients; CTC forced alignment; installed codec interface |
| Retained vision integration | 9 | CPU image codecs/transforms, HIP transfer and synthetic ResNet18 gradients |
| Retained core | 7 | GEMM/math SDPA/normalization/FFT output/backward and RNG replay, consumer imports |

Integer PCM fixture results and saved PCM samples match exactly. MP3 output
matches the separate FFmpeg CLI interface exactly in the owned fixture; this is
not an independent mathematical MP3 decoder implementation. GPU resampling uses
CPU float64 output/weighted input-gradient references with rtol0.002/atol3e-5;
WAV quantization permits at most1/32768 absolute error. Native/DSP gates retain
their original precision limits. Ten600-second guards acknowledged; each GPU
gate and both complete passing batches have identical before/after kernel logs.

Mapped codec identities include ten TorchCodec/FFmpeg shared libraries, FFmpeg
`6.1.1-3ubuntu5`, libavcodec60.31.102/libavformat60.16.100/libavutil58.29.100.
Only this pinned CPU audio scope is qualified. Video/images, GPU codecs, all
formats/containers/metadata, arbitrary models, long streaming, file-I/O gradients,
universal stability and performance are unqualified. The historical unknown
Python CPU fault remains unresolved and distinct from the previously repaired
VM-heap/CSR defect and B04.2's demonstrated cold-hostcall queue path.

Initial failures are retained: user app-document preservation refusals,
missing ROCm CMake path, a competing UID1000 `train` preflight refusal and missing
native architecture configuration. No failed GPU numerical gate was hidden or
relaxed. The train process exited; the user confirmed the board was free. The
corrected build-r5 and candidate-r1/installed-r3 all pass.

The user explicitly requested keeping `finetune-n7` out of this work. Eight
historical app identities were excluded without further app inspection; the
shared qualifier has an explicit `include_training_app=False` mode for this
stack-only batch. Its old default remains for historical callers. The preserved
306 stack-owned identities, four FFT AOT databases and compatibility links match.
No application qualification or checkpoint migration is claimed.

## Recovery and evidence

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Only active development runtime JSON
image/codec metadata changed; existing ROCm/Torch DSOs and kernel packs remain,
including HIP SHA256
`a1162308076a99ff5cf5e4e15ca8bff53928d8bc4e624c42033d7e15dd2d1e3b`.
The B04.2 CLR queue repair stays installed. New codec libraries live inside the
new image. Old images/releases/caches/checkpoints/source history remain.

Recovery: `WS/recovery/codec-b043-before/runtime.json`; the original shared
qualification helper is also retained there. Run as root under the exclusive
runtime lock to restore the exact previous runtime metadata/image:

```sh
python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/update_dev_prefix.py -- python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/deploy_codec_b043_on_host.py --rollback
```

Rollback was not used; installation automatically restores the previous metadata
if its installed gates fail. Recovery does not remove any image or build.

Raw evidence `WS/logs/codec-b043-*` is SHA-verified locally in
`kernel/artifacts/codec-b043/remote`:145 raw files/3,703,497 bytes, plus wheel,
source archive/provenance, Git comparison, build package inventories and recovery.
The durable export archive SHA256 is
`b52252d23c892c260c9aa82cc6a321965df4ac93aea7acf5e30ea3f67efa4323`.
Current baseline: `WS/logs/codec-b043-qualification-r3/final.json`.

Entry had a new boot ID `8ea98081-ca52-4869-ae8f-cf8072362d80`, still
`7.1.5-bc250rocm-r2`; it remained the same throughout the passing gates. No agent
reboot, BIOS/bootloader/fallback change, .105 write or deletion occurred. Disk
free34,926,919,680 bytes observed; no cleanup by this task. No own running build,
test, container, guard or export; root SSH sessions closed.

**Next B04.4:** inspect the pinned CPU video APIs and qualify owned raw-frame
video decode/seek/encode references with explicit HIP transfer. CUDA/image codec
builds stay off until separate qualification; existing torchvision image support
remains available. Broader B01/B02/B03/B05/B06/B07/N2–N8 remain open.
