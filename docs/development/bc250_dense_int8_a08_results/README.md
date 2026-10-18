# A08: native dense INT8 to INT32 — 2026-10-18

The existing gfx1013 hipBLASLt scalar backend now accepts signed INT8 A/B,
INT32 C/D, `COMPUTE_32I` and INT32 alpha/beta. The installed patched Torch already
called this API, so `torch._int_mm` now works through its ordinary GPU route
without a Torch rebuild or launcher hook. This applies to callers using the
selected .104 ROCm prefix, not just the LoRA trainer.

Library source `319a235d6c6496f0da37be44dfa60eaf31a29f82`, TheRock runtime helpers
`8f0dfb671e88b31bc3f5bb4a4958acefdafde556`. Torch remains
`4af4a1508a6ec0b7b2b29f5e0115f8477c4d1bb5`; A09 profiling remains enabled.

## Implemented contract

A scalar tiled HIP kernel loads signed bytes, multiplies in INT32 and accumulates
in unsigned32. This explicitly defines modulo-2^32 overflow without signed C++
overflow or FP32 rounding. Alpha/beta/bias arithmetic also wraps modulo32; RELU
applies to the resulting signed value. There is no saturation or tensor-core claim.

- Row/column layouts, N/T transpose, padded leading dimensions, matrix offsets,
  strided batches and zero-stride input broadcasts follow existing contracts.
- Host/device **scalar** INT32 alpha/beta, zero-alpha/zero-beta no-read behavior,
  exact C/D aliasing with matching layouts, optional per-row INT32 bias with
  broadcast/strided batches; DEFAULT, BIAS, RELU and RELU_BIAS epilogues.
- Alignment and overlapping spans are checked for integer inputs/output, C, bias
  and device scalars. Output batches must not overlap. Device-scalar mode requires
  valid A/B/C pointers conservatively because scalar values are not read on CPU.
- Other types/scales/compute modes, pointer vectors, nonlinear/aux epilogues and
  mixed outputs remain refused. Public descriptor creation still rejects the
  existing INT32 pedantic mode; it was not added by this bounded task.

The earlier probe requested that unavailable pedantic descriptor mode and stopped
before matmul dispatch. It is retained as [initial refusal evidence](initial-refusal.log).
The supported plain COMPUTE_32I cases pass; no numerical tolerance was relaxed.

## Numerical and integration evidence

[99 direct API cases and11 refusal contracts](direct.log) pass exact CPU references:
96 padded/batched/order/transpose/scalar/epilogue combinations, positive/negative
accumulation overflow, and127*127*1041 =16,790,289 (odd and above FP32's exact
integer range). CPU sums use INT64 before explicit modulo output/scalar arithmetic.
Output padding/sentinels are compared as well. Zero scalar tests pass null
unread inputs; nondefault streams and heuristic/implicit algorithms are covered.

[16 public Torch configurations](integer.json) compare exact CPU INT64 matmul
converted to INT32: row/column/strided inputs, sizes17–128, K8–131080, custom
streams and `out=` (32 matmul calls). Five public refusal checks retain existing
Torch shape/type/output constraints. The [profiler trace](trace.json) records the
native `gemm_int8` GPU kernel, not a host or floating fallback.

Existing [floating Lt suites](float-regressions.json) pass403 scalar +561 extended
+664 skinny cases. [Static checks](static-checks.json) confirm unchanged token
streams for the old floating gemm and launch functions; validator type gates
were extended. Clang-format and six Python helper format checks pass. The small
probe includes an unused legacy renamed-main helper that emits a pre-existing
missing-return warning; no warning suppression was added.

[Qualification](qualification.json) also passes core7, profiler7, sparse backward38,
convolution30, A06=42, A07=36 and three actual Qwen3.5-2B train/checkpoint/nonzero-
adapter comparisons. A06 BF16 gradients remain exact for all372 adapters at65/1024;
A07 retains the recorded2.17%/1.31% worst-adapter bounds and12 head256 calls.
User checkpoints/datasets were not used or changed.

## Bounded timing

Five alternating pairs, ten eager calls per HIP-event envelope after two warmups;
resident byte/FP32 inputs and preallocated outputs. The baseline is FP32 GEMM
plus conversion into INT32, with an extra kernel and resident float operands.
Its small bounded sums are exact; it is not a complete quantization pipeline or
kernel-only timing. [Raw samples/protocol](benchmark.json):

| M=N=K | INT8 ms | FP32 GEMM + conversion ms |
| --- | ---: | ---: |
|32 |0.01473 |0.02179 |
|128 |0.01528 |0.03440 |
|256 |0.05640 |0.05550 |

Results are mixed and narrowly scoped. No universal performance, LoRA throughput
or quantized-model speedup is established. Dense FP32/BF16 defaults are unchanged.

## Usage and recovery

```python
import torch

a = torch.randint(-128, 128, (32, 64), device="cuda", dtype=torch.int8)
b = torch.randint(-128, 128, (64, 32), device="cuda", dtype=torch.int8)
result = torch._int_mm(a, b)  # INT32 GPU tensor
```

The pinned Torch API retains M>16, positive K/N divisible by8 and contiguous
INT32 output constraints. Raw Lt's separately tested shape/layout contract is
broader. Inputs must already be quantized by the caller; this does not install
bitsandbytes/torchao, QLoRA, packed low-bit formats or integer autograd.
Compile/graph integration and large quantized models remain unqualified.

Targeted `hipBLASLt+dist` builds used ccache20GiB/jobs12 under the exclusive prefix
lock. [Build](build.json), [source and final verification](final.json) preserve
exact source/DSO/external-kpack hashes.61 protected hashes, including current and
old Torch/runtime/SDK/app files and the frozen old Lt pair, remain unchanged.
Both Lt DSO and kpack must be restored together under `update_dev_prefix.py` from
`WS/recovery/dense-int8-a08-before`; original source and runtime metadata are also
backed up. Guarded qualification restores the old pair/source on failure.

[12 guards](guards.json) all acknowledged600s deadlines and have unchanged per-run
dmesg. One failed probe was the retained pedantic descriptor request. No agent
GPU job/container/guard deadline remains. Provenance is
`WS/SOURCE_DENSE_INT8_A08.json` and the current prefix copy; raw logs are
`WS/logs/dense-int8-a08*` and ignored local `kernel/artifacts/dense-int8-a08/remote-logs/`.
Full Git, patch0031 and verified incremental bundle onfcf7a714 preserve source.
Old sources/builds/caches/images/symlinks and user-owned src remain; no kernel,
BIOS/boot, .105, SDMA0/heap-off or default BLAS policy change. Disk~1.10GiB free;
check capacity before subsequent builds. Historical Python CPU segfault remains
unresolved and distinct from the fixed VM-heap/CSR corruption.

A01–A09 are complete within their recorded bounded scopes. Optional quantization,
remaining attention/profiling contracts and B-items stay separate. Next is N2.7:
resume the existing small Schur QR/device-iteration work with preserved reference/
failure gates and host-LAPACK production default.
