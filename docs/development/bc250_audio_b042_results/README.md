# B04.2: native torchaudio on gfx1013

Torchaudio **2.11.0+gfx1013 is installed** in the development runtime on .104.
Existing patched Torch2.13.0a0/HIP7.17.0/Python3.12/ABI1 is retained. GPU signal
operations and bounded first-order backward checks pass; audio file loading and
saving still require the absent TorchCodec backend. Next task **B04.3** addresses
that CPU codec integration.

Use the installed package through the existing `bc250-torch` or `bc250-train`
runtime launchers. Existing host virtual environments/release packages were not
rewritten. For example:

```sh
bc250-torch -c 'import torch, torchaudio; x=torch.randn(1,1600,device="cuda",requires_grad=True); y=torchaudio.functional.resample(x,16000,8000); y.square().mean().backward(); torch.cuda.synchronize(); print(torchaudio.__version__,y.device,y.shape,x.grad.isfinite().all().item())'
```

This example uses GPU tensors. It makes no file decoding or speed claim.
`torchaudio.load` and `torchaudio.save` currently refuse with the explicit missing
TorchCodec requirement. CUDA CTC decoder remains disabled; GPU media decoding,
BF16 custom operators, arbitrary models and second derivatives are unqualified.

## Implementation and revisions

| Component | Preserved source / installed identity |
| --- | --- |
| Audio upstream | v2.11.0, `34c52a67e8941bbd8e6adaca0eb0b9eabec11d78` |
| Audio port | `sources/bc250-torchaudio`, branch `users/Naster17/gfx1013-audio`, `1ae7f5d8023e82be51b500399a37fceb45481a2b` |
| ROCm systems | `1f9609fd69c4923291f58934495520fe62e66ef8` |
| Integration / runtime controllers | `d75350905` / `7bf89b571` |
| Installed image | `97fac19f0d5747265c4ff8328f99d2eaacaf8ff62e5092d22cda300f7130a632` |
| Installed HIP SHA256 | `a1162308076a99ff5cf5e4e15ca8bff53928d8bc4e624c42033d7e15dd2d1e3b` |
| Runtime manifest SHA256 | `5f2ff6801ead12218188b53571973ba3e021c9b8bc70f4a1016eebca06e4467b` |

Four reviewable audio patches under `patches/amd-mainline/torchaudio/` enable
ROCm extension sources, launch IIR on the current stream outside the hipify
sensitive dispatch macro, fix RNNT device-index guarding and order RNNT counter
resets on the consuming stream. No replacement Torch wheel was installed.

Qualification exposed a reproducible **CPU crash in HIP cold hostcall queue
initialization**. Original HIP `.text` was verified byte-identical to a relinked
symbol-bearing diagnostic library. Its call chain reaches
`getOrCreateHostcallBuffer` through an internal synchronous bitfield fill. A
reclaimed transfer queue had no hardware queue; the direct blit bypassed normal
queue reacquisition. The CLR patch locks transfer execution, reacquires/checks
the hardware queue and holds that lock through the fill. The original full RNNT
sequence passes with the repaired runtime; the counter reset correction alone
still failed. Scope is this demonstrated path. The separate historical Python
CPU execution fault remains unresolved, distinct from the already repaired
experimental VM-heap/CSR corruption.

Runtime base `62ab2437c`, native Torch `dc8fea0`, Torch source `2b449c4`, rocBLAS
`8389197c` and other libraries `319a235d6` remain the base; **HIP has the new CLR
delta above**. N3.2/math SDPA/SDMA0/experimental-heap-off defaults remain. No
external matching HIP kpack was found; other libraries' kernel packs remain.
Content-bound checkpoint provenance can detect the changed HIP identity; no
checkpoint migration or trainer edit was performed.

## Evidence and limits

[Machine-readable summary](summary.json) contains identities, counts, numerical
errors, native GPU trace names and result details. Counts are composite checks
with distinct scopes, not an API completion percentage.

| Gate | Checks | Reference / scope |
| --- | ---: | --- |
| DSP | 34 | FP32/64 spectrogram, mel, MFCC, resampling, deltas, gain, preemphasis; contiguous/strided outputs and input gradients; explicit CPU DFT oracle |
| Native audio | 40 | IIR orders1/3, waveform/a/b gradients, default/nondefault streams; profiled native IIR replay; FP16/32 RNNT costs/logits gradients against CPU FP64 dynamic programming and enumerated tiny paths; forced alignment exact CTC paths/score references; explicit missing codec refusal |
| Retained integration | 67 | Vision operators51, vision integration9, core7 |
| Direct HIP queue probe | 1 | 3 rounds of20 created/warmed/cold-hostcall-initialized/destroyed streams;384 CPU-reference/guard words |
| Candidate total | 142 | Main141 plus separate direct HIP probe |
| Installed total | 142 | All above against active runtime |
| After user reboot | 48 | Native audio40, core7, direct probe1 |

Maximum relative L2: DSP `3.7622219068142665e-6`, native
`3.3461023213142644e-4`. Original precision limits were unchanged. IIR profile
contains actual GPU forward/backward kernels on the consuming stream. The direct
probe retains cold hostcall metadata without triggering actual device
printf/assert: this does not qualify the printf service. No all-shape/type,
universal stability, convergence or throughput claim follows from these tests.

All26 600-second guards acknowledged. Passing candidate-r4, installed-r5 and
postboot-r6 batches each have identical before/after kernel logs. The whole
session log differs: early RNNT CPU faults and a separate concurrent global host
OOM are retained. The OOM killed an external Qwen1.5B Vulkan workload and own
small staging controllers; exclusive causality was not established. After user
reservation, resource/GPU-client preflight was added and subsequent gates pass.
Failed hipify builds, original RNNT failures and seven GDB diagnostics remain.

The user confirmed rebooting during evidence copying. Boot changed from
`efd16227-e452-4939-acbc-a6db47673788` to
`c81853b2-fba0-4e30-ab67-0c7e06f221bd`, still `7.1.5-bc250rocm-r2`. Installed
identities match and all48 postboot checks pass. No agent reboot, bootloader,
BIOS, fallback or .105 changes occurred.

## Preservation, recovery and reproducibility

`WS=/home/nik/bc250work/rocm-10.2-gfx1013` on .104. Final preservation baseline:
`WS/logs/audio-b042-qualification-r6/final.json`,314 protected paths and four FFT
AOT databases/compatibility links retained. One changed entry app document was
reviewed/preserved without renewed trainer qualification. Previous image
`a0b85fa88c9ecfbec88a8774f19db0fbc18fcfeaa086a4e402b69aa99ba55e27`, sources,
builds, caches, wheels, recovery and user data remain.

Recovery directory: `WS/recovery/audio-b042-before`. The deployment helper
restores original runtime metadata/image, HIP DSO and exact CLR source together,
checking expected identities. Run as root through the exclusive prefix lock:

```sh
python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/update_dev_prefix.py -- python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/deploy_audio_b042_on_host.py --rollback
```

Rollback was not used. Ordinary launchers retain their shared runtime lock.

Audio build: four Ninja jobs,10GiB memory/12GiB memory+swap,8 allocated CPUs,
20GiB shared ccache with explicit HIP compiler caching; automatic CPU
cpp-extension wrapping bypassed caching for initial CPU units. Runtime repair:
one isolated translation unit/archive replacement/link,6GiB memory/8GiB
memory+swap,4 CPUs, shared cache. Candidate artifacts were built separately and
mounted read-only for testing before locked installation.

Durable local `kernel/artifacts/audio-b042/` holds initial Git-bearing source
archive plus incremental bundles/patches, both raw wheels, CLR patch, recovered
runtime/source/native probe, symbolication proof and301 raw log files
(9,240,283 bytes). Initial zero-byte reboot-interrupted evidence-r1 remains;
fsynced evidence-r2 was copied and SHA-verified successfully. Archive SHA256:
`0a255dbd38725ec4ef71bc7e599a826f883fdb8fb95311c81425ce5af1c15670`.
Installed wheel SHA256:
`4ac9faa872e4138be0f7242f0c4a988ca63a4eb51df3562c7096e69a0b901328`.
Raw remote evidence: `WS/logs/audio-b042-*`. Native payload and old recovery are
also exported locally; source Git histories and four audio/one CLR patches are
preserved for future repositories.

No own build/test/container/guard/export is running; root SSH sessions closed.
Last observed disk free17,554,157,568 bytes. **Next B04.3:** inspect pinned
TorchCodec ABI/FFmpeg requirements, build separately, qualify owned CPU audio
file load/save references before installation. Preserve this native signal path.
