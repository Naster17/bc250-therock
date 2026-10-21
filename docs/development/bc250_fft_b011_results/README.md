# B01.1: explicit BF16 FFT promotion on gfx1013

**Installed opt-in helper:** `bc250_fft`0.1.0. BF16 input is converted to FP32 on
its existing device, then passed to the existing native `torch.fft` operator.
No CPU fallback or global monkeypatch; ordinary `torch.fft` behavior is unchanged.
Without `out=`, results are complex64 or float32 according to the transform.
Autograd casts input gradients back to the leaf dtype, including BF16 rounding.
This uses FP32 computation and an extra conversion allocation, not native BF16
FFT storage or a speed optimization.

Use in the development runtime through `bc250-torch` or `bc250-train`:

```python
import torch
import bc250_fft as fft

x = torch.randn(5, 7, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = fft.rfft2(x, norm="ortho")  # GPU complex64; no CPU fallback
(y.real.square() + y.imag.square()).mean().backward()
assert y.is_cuda and x.grad.dtype == torch.bfloat16
```

The helper exports the18 complex/real/Hermitian forward/inverse1D/2D/nD transforms.
Other dtypes/arguments are forwarded to the native operator. `out=` keeps the
native output dtype/layout/error semantics; it does not add an autograd promise
or force an explicitly supplied output tensor to complex64/float32. Existing
FP16 restrictions remain native. Native torch.fft BF16 calls still reject, both
outside and inside GPU BF16 autocast. Applications must import/use this helper;
no user application was modified, inspected or tested.

## Audit and implementation

The qualified source already promotes integer FFT inputs and accepts float32/
float64 plus selected device-half inputs. It rejects BF16 in `SpectralOps.cpp`
and the FFT decompositions. The pinned autocast table registers FFT FP32 policy
on CPU, not GPU. Runtime audit confirms BF16 rfft2 rejection in both GPU AMP
contexts and successful explicit float conversion/output/backward. No prior
stack helper was found; N2.14 already qualifies native FFT plans/layouts and
therefore was not reimplemented.

Source [bc250_fft.py](../../../build_tools/bc250/runtime/bc250_fft.py), integration
`427492188`, helper SHA256
`50bada1588a98dc2a18eb5948a231a76ec7aaede8fe6198b503c338ef9820fe6`.
A pure Python module layer was added to an independent image; no Torch/native
ROCm rebuild, ccache operation or build GPU devices. The first dynamic-export
candidate passed327 checks; explicit export assignments replaced it for static
lint/API visibility, followed by a complete unchanged final-candidate replay.
An intervening staging-r2 FileExistsError on an already-owned tmp directory was
corrected with exist_ok; failed artifact directories/logs remain. No GPU
numerical case or tolerance was removed or relaxed.

## Evidence and bounds

[summary.json](summary.json) contains pins, counts, maxima, original audit and all
helper results. **327 final-candidate +327 installed composite checks pass**;
a separate3-check audit and earlier327-check candidate also pass.

| Scope | Checks per candidate/installed replay | Reference |
| --- | ---: | --- |
| BF16 transforms | 216 |18 operations ×3 norms ×2 layouts ×2 GPU AMP contexts; odd/prime crop/pad, nonadjacent dimensions, strided offset leaf/unused storage; CPU float64/complex128 output and weighted all-leaf gradients |
| Non-BF16 | 20 |18 FP32 operator pass-through plus FP64/complex64 controls; exact native comparison |
| Contracts | 5 | Native norm/dimension/type/autograd-out refusals and unchanged direct BF16 rejection in both AMP contexts |
| Output storage | 2 | Complex and real strided out with sentinel gaps intact |
| Independent DFT/gradient | 1 | Explicit17-point CPU DFT and three continuous-reference finite differences, GPU output and cast-back BF16 leaf gradient |
| Native GPU trace | 1 | Fresh adapter forward/backward, CPU reference and named rocFFT kernels |
| Retained native FFT/core | 82 | N2.14 real72/contracts3 and core7 |

Output limits rtol=atol5e-4. BF16 leaf gradients compare against CPU double before
rounding: rtol.02/atol2e-5 and relativeL2<.004, with unused leaf elements exactly
zero. Maximum output absolute error6.4036e-7, leaf-gradient relativeL2=.0019803
(about0.20%). These are fixture maxima, not universal accuracy guarantees.
Continuous-reference finite-difference errors<=1.336e-12; this verifies the
continuous DFT derivative, not differentiation through quantization as a discrete
function. Trace has32 GPU kernels including named length17 rocFFT forward and
backward. No CPU dispatch substitute was accepted.

Thirteen600-second guards acknowledged across the audit/three full replays.
Every GPU gate/whole passing batch has identical before/after kernel logs.
306 stack hashes/four existing FFT AOT databases/links retained; process-local
plan reuse/capacity guards pass. Source/native Torch/ROCm/defaults/math SDPA/
SDMA0/heap-off unchanged. No all-shape/plan/device, compiled/graph, higher-order
backward, native BF16 FFT, performance or sustained-stability claim. The unrelated
historical Python CPU fault remains unresolved/distinct from VM-heap/CSR and
B04.2's demonstrated queue repair.

## Recovery and handoff

Image `5d0a6d9cef758b2f7677a2b52b9d29ac51e24970703259a10ca8e44bee93feb1`;
runtime JSON SHA256
`ab9db860207e230ea704cab2bf76005d19b7c8967e59559a5a4b0088ed7f38c1`.
Only image/module metadata changed in the development prefix. Prior image
`94c610368d0fcf596a9f9f6355a5d5d28fd1111d54d4033c46e6b5977cae1ece`, existing
sources/builds/caches/checkpoints/releases and compatibility links remain.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`; recovery
`WS/recovery/fft-b011-before/runtime.json`, plus original shared qualifier.
Run as root under the exclusive prefix lock to restore exact prior metadata/image:

```sh
python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/update_dev_prefix.py -- python3 /home/nik/bc250work/rocm-10.2-gfx1013/helpers/deploy_fft_b011_on_host.py --rollback
```

Rollback unused; failed installed gates would restore prior metadata automatically.
Active baseline `WS/logs/fft-b011-qualification-r3/final.json`.
Local `kernel/artifacts/fft-b011` retains final/prototype module, controller variant,
recovery and141 SHA-verified raw log files4,140,885 bytes plus13 guard states.
Remote raw `WS/logs/fft-b011-*`. Durable export SHA256:
`77cf1bd74524c66727f514a7ead1ebc50469b21f51021bd82a4947430d722f4a`.
Same r2 bootc646d874; no agent reboot/.105/BIOS/fallback/deletion change. Disk
21,439,528,960 bytes free observed. No own running build/test/container/guard/
export; root SSH sessions closed. User `finetune-n7` remains excluded.

**Next B02.1:** audit general sparse-solve/factorization consumers and existing
N2.13 triangular evidence. Select feasible native ROCm integration or an explicit
bounded fallback with memory caps, residual/singularity and promised-gradient
reference contracts before implementing it. Native BF16 FFT is not relabeled as
implemented by this explicit FP32 fallback.
