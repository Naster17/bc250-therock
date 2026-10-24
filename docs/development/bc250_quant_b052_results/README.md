# B05.2: native weight-only INT8 layout correction

The daily `.104` Torch build now handles strided activations and FP32 row scales
correctly in `_weight_int8pack_mm`. This fixes the wrong results reproduced in
B05.1 without adding quantization/model routing or changing native arithmetic.
The opt-in RCCL profile remains on its older build and **does not yet contain
this fix**. `.105` was not contacted. [Verified identities](verification.json).

Torch source:`a5249802372882dedccb91d3fd4d73be9fcdf2bd`.
Tooling:`506ae3b9e797f766b1d9fed56916006499fc583c`.

## Change and build provenance

The generic HIP kernel indexes activations/scales as contiguous row-major
arrays. `.to(float)` preserves transpose layouts and leaves FP32 views unchanged.
The fix adds `.contiguous()` to both converted operands, matching the existing
weight handling. The generated `int8mm.hip` differs from the qualified daily
source by exactly these two calls. Accumulation, output cast and kernel math
are unchanged. Normal contiguous calls incur no extra layout copy.

The local qualified `torch/` source retains its previous RCCL guard history.
An independent `WS/pytorch-quant-b052` copies the daily2b449c4 source/build, with
the complete six-file tracked source delta througha524980 applied before pip.
Existing third-party dependencies are mounted read-only; no reset, source-tree
replacement or hardlinks to writable active native files are used.

Build uses only `python -m pip install -e . -v --no-build-isolation` through the
pinned28ad36e6 build image: native gfx1013, NCCLoff/Glooon, four CPUs/four jobs,
one linker,10GiB RAM/12GiB memory+swap, shared20GiB ccache without reset, no GPU
or network. The initial~8GiB free is sufficient for the~3GiB independent copy;
post-copy disk floor3GiB is checked. Existing source/build/cache history remains.
Cumulative ccache statistics are not build timing or newly introduced failures.

Pip exits0. The first verifier incorrectly requires raw source equality after
Torch's automatic HIP generation and stops on the NCCL files. Both translated
files match the protected B03.7 versions exactly; they are disabled in this
NCCL-off candidate. All other overlay hashes match. The verifier is corrected to
record pre/post-generation identities, and the original failure is retained.
A subsequent seal check initially expects CMake `OFF/ON` spelling, while this
cache stores `0/1`; that metadata check is corrected. The early qualification
controller refuses the unsealed build before any GPU test. No compiler/native
build retry, numerical tolerance relaxation or backend workaround is involved.

## Qualification and installed behavior

**947 candidate checks across11 gates**, followed by **231 ordinary-launch
checks across three gates**:

- Native220:216 exact CPU-double comparisons across FP32/FP16/BF16, two odd/tail
  geometries, four activation layouts, three scale layouts and three byte-weight
  layouts; output dtype/device/shape and full operand/padding guards. Private
  stream handoff, backend identity and retained backward/INT4 refusals also pass.
- W8 recipe71, core7, sparse253, FFT245, compiled dense14/CNN14, profiler1,
  CPU Gloo31, vision51 and native audio40 pass using existing references and
  tolerances. Compiled model gates do not establish compiled quantized-op support.
- After exclusive-lock activation, public `bc250-torch` repeats native220;
  public `bc250-train` checks frozen consumers/daily backend5; explicit
  `rccl-single` launch6 confirms the unchanged older profile still loads.
-402 protected hashes/four AOT DBs/links match the installed baseline. The daily
  Torch target/runtime metadata are deliberate changes; old native/source files
  and the byte-exact manifest backup remain pinned. Fourteen600s guards ACK;
  accepted gate/candidate-batch/final ordinary kernel logs are identical. Same
  r2 boot5c04afbb-ff2f-4f4b-9eac-8238bb9288fc, no reboot.
- Torch device-generic18-configuration regression is added; targeted eight
  spin/lintrunner checks pass after removing an existing unused test import.
  The full upstream test suite/other GPU hardware were not run.29 local BC250
  tests and selected helper Ruff checks/format pass.

Daily `/opt/bc250-stack/current/torch` selects `WS/pytorch-quant-b052`;
manifestSHA`561f014e31c4ca0bba73861d56125e4e18124867904b7c23cff8d7e0fa36d2a1`.
Native libtorch_hip SHA`2f2761cd7f259612863406ab0e85d7b1b5ce8b59a32655bf9ac7db9359d0df27`.
Image506f3b26, HIPa1162308, ROCm libraries/packs, N3.2 tuning, math SDPA,
SDMA0/experimental-heap-off defaults and Gloo/NCCL-off selection are retained.

No script change is needed for callers of the **daily** native operator.
The W8 helper remains explicitly imported/not installed. Native input backward
still raises its existing derivative error; packed INT4 still rejects gfx1013.
The older opt-in RCCL Torch14cf/profileSHA42eb7631 remains untouched, including
its known native layout defect; the W8 wrapper's normalization remains useful
there. No universal numerical/API/performance, peak-memory, model quality,
training speed, higher derivative, multi-device or broad stability claim follows.
The historical Python CPU fault remains unresolved/distinct from fixed
VM-heap/CSR/hostcall defects.

## Recovery and next action

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Raw `WS/logs/quant-b052-*` includes
build/failed checks/candidate/ordinary evidence. Main candidate final:
`WS/logs/quant-b052-qualification-r1/final.json`; installed final:
`WS/logs/quant-b052-ordinary.json`. Candidate release:
`WS/quant-b052/candidate-release-r1`. Old daily source/build remains
`WS/pytorch-distributed-b031`; old manifest:
`WS/recovery/quant-b052-before/runtime.json` (SHA32fd4559).

Identity-guarded rollback restores the old Torch link and exact manifest. It was
not exercised because all acceptance gates pass:

```sh
WS=/home/nik/bc250work/rocm-10.2-gfx1013
doas python3 "$WS/helpers/update_dev_prefix.py" -- \
  python3 "$WS/helpers/deploy_quant_b052_on_host.py" --rollback
```

Native candidate build/staging/activation/export helpers run under the daily
exclusive update lock. Candidate gates use the existing internal harness rather
than nesting ordinary-launcher shared locks. Public installed acceptance runs
outside the exclusive lock and holds the normal shared lock per invocation.
No user application, old cache/build, fallback or BIOS was modified/deleted.

**Next B05.3:** check disk/resources and apply the same native layout correction
to an independent RCCL-enabled Torch candidate, qualify its exact layout and
retained RCCL/gradient contracts, then replace the opt-in profile only under the
daily lock with its own matched identity update/ordinary acceptance/rollback.
Keep the new daily Torch and `.105` unchanged. Broad low-bit model integration
and W8 performance work remain separate.

Local `kernel/artifacts/quant-b052` preserves the verified193,263,147-byte archive
SHA256`fce8095e5c163f36d100f9730137faad0fcd06b98014448e1f17db729a002288`,
153 raw log files/5,516,695bytes, native DSOs/extensions, generated source and
CMake provenance, original overlay, eight tools, recovery and14 guard records.
All eight helper hashes and nine native DSO hashes match the extracted payload.
No own build/test/container or armed guard remains at handoff.
