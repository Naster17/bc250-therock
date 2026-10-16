# A01: convolution timing and workspace — 2026-10-16

**Implemented, installed on .104 and bounded-qualified.** Source/helper revision
`380a403a1bc17fa561345846fb4d92a6f57c3237`; ROCm systems
`329eaf726632ab0136924804cbc1321ae09fc126`, libraries
`08ba6f386ff661e720873779fc417b4246787d9c`. Torch4ee6571, Triton ee65ff3b,
image5df0a58, kernel0957f1/image84995f1 and boot a287fd16 remain unchanged.
The frozen20261013 software fallback and r2 boot fallback remain intact.

## Demonstrated causes and corrections

The original B2/C4/L19, W6x4x3, stride2/pad2/dilation2 failure reproduced through
MIOpen's public C API, independently of Torch convolution dispatch. Baseline3/6
cases passed; adding4096 bytes of workspace still failed forward and dx selection.
An unchanged-call HIP observer captured successful `hipEventElapsedTime` calls
returning negative durations. Separate event bracketing and disabling direct
dispatch did not reliably repair it. A smaller event-only fixture found32/32
completed event pairs unstable/nonpositive on repeated reads, even while retaining
the same event objects. These are timing failures, not evidence of wrong convolution
arithmetic or the historical Python CPU segfault.

The read-only KFD/shader probe measures two approximately100MHz counters with
starting points about14,360seconds apart on this boot. Kernel gfx_v10_0 uses
Cyan Skillfish's SMUIO GOLDEN_TSC, whereas shader `s_memrealtime` has a different
epoch. ROCr extrapolated this offset through its clock-drift calibration ratio,
amplifying small calibration changes into large event-timing errors. A diagnostic
HIP build recorded live/cached timestamp disagreement; forcing extra command
completion stalled in userspace and was terminated, exit137/guard ACK/no kernel
change. That diagnostic source/object was discarded and the original qualified
HIP source rebuilt without installing a HIP change.

ROCr now applies its existing Windows epoch-offset mechanism to native gfx1013,
under the existing calibration mutex. Linux detection requires a completed
dispatch translated more than one second into the future. Other Linux GPUs keep
the old path; Windows retains its threshold. No architecture spoof, timing clamp,
fabricated positive duration, solver-forcing flag or host-compute substitution.
The first query anchors the offset; **absolute CPU/GPU trace alignment is not
qualified**, and query latency can bias that anchor. Opposite/negative epoch
offsets, reset/reboot transitions and other boards remain unqualified.

Separately, MIOpen's workspace query rejected every dilated GEMM convolution,
while Find offered applicable im2col/col2im GEMM solvers. Removed that obsolete
exclusion and retained solver applicability/size checks. Cold normal search now
queries480 bytes for the original fixture and evaluates both Direct/GEMM with
CPU references. Warm hybrid/fast selection can legitimately query0 for its chosen
zero-workspace solution. Insufficient-workspace validation remains enabled.

## Qualification

| Gate | Result and scope |
| --- | --- |
| Isolated ROCr event candidate |32/32 positive, stable completed-event pairs;160 timing reads, default/custom streams |
| Isolated ROCr with original MIOpen |15/15 direct API configurations pass CPU-double fwd/dx/dw; queried/extra workspace and stream/profiling variants |
| Combined candidate, fresh private FindDb |15/15 direct API configurations; loaded-library maps prove one candidate MIOpen instance |
| Torch candidate |30/30 output/requested-gradient configurations: FP32/FP16/BF16, Conv1D/2D/3D groups1/2, dilated/strided, channels-last/noncontiguous, inference/dx-only/dw-only/both, benchmark/deterministic controls and transpose-conv |
| Installed normal runtime |Same15 direct +30 Torch configurations pass; fresh private FindDbs; no diagnostic preload/path override |
| Installed timing |32/32 stable event pairs plus six1/2/5ms raw shader-counter references on default/custom streams; repeats after100ms unchanged; event intervals exceed kernel-only intervals by about10–18us |
| Installed core regression |Seven prior bounded FP32/BF16 GEMM, math SDPA D32/D256, layernorm/FFT gradients and exact RNG-replay groups, plus pinned consumer imports |

These are overlapping bounded gates, not a universal coverage percentage or a
performance improvement. Earlier GPU-event benchmark claims (including N1.8)
require remeasurement before relying on their speed ratios; numerical results
remain separate. Host-wall-time results are not automatically invalidated.

All21 executed GPU/diagnostic gates have identical before/after kernel logs and
600s guards ACKed; `gates.json` records every exit, including expected defects and
harness failures. Collector exit0 on baseline probes is not all-cases-pass.
Retained failed attempts: bracketing/direct-dispatch hypotheses; terminated extra
completion diagnostic; two uninstalled-ROCr loader failures before adding its
sysdeps search path; initial fresh-DB permission failure (then corrected with a
new label/directory). No core, driver fault, reset or reboot occurred.

`deploy_miopen_a01_on_host.py` ran under the exclusive runtime lock, staged only
ROCr/MIOpen install components with their installed RPATHs, atomically replaced
the two versioned DSOs, ran five installed gates and verified seven protected
runtime/fallback hashes. It restores both original DSOs and runtime metadata on
failure. Copies, never dist hardlinks, were retained at
`WS/recovery/miopen-a01-before`; installed/source identities are in
`WS/SOURCE_MIOPEN_A01.json`, also copied into the current development prefix.
Targeted16-job builds used the existing20GiB ccache; no GPU devices were mapped
into builders. No global build, cache wipe or package installation.

| Installed DSO | SHA256 |
| --- | --- |
| libhsa-runtime64.so.1.21.0 |1e4c4d58f4e2424ebb013238896ad024355727e720e4b44619fb11ef1d5cd228 |
| libMIOpen.so.1.0 |85a41de0c8cda754340f037bf978ef72067047e25a70c3be3217ed6c04dd83ab |

Full raw logs remain at .104 `WS/logs/miopen-a01-*` and locally under ignored
`kernel/artifacts/miopen-a01/host-logs`. This directory commits result/guard/exit
records, representative failing traces, build logs and patch reconstruction.
Library27-patch/system4-patch trees reconstruct exactly. The broader legacy P2
reconstruction helper still misses three later SDPA files in current Torch4ee6571;
its overall failure is retained as a separate packaging gap, not hidden or caused
by A01. The complete qualified Torch Git history/source remains preserved.

Changes apply to programs loading the **current development prefix**, including
bc250-torch/bc250-train; unrelated host venvs or other ROCm prefixes are outside
this deployment. Existing real user data/checkpoints/caches and untracked src/
remain intact; .105, BIOS, boot policy, SDMA/heap-off/math defaults unchanged.
The historical Python CPU fault remains unresolved, distinct from fixed VM-heap/
CSR corruption. No agent build/test/container or recovery deadline remains.

**Exact next: A02**, the separate MIOpen bundled legacy CK reduction gfx1013
feature/configuration path, with direct reduction and LSTM forward/backward CPU
references. Missing optional CK plugin/system FindDb/KDB warnings remain A03/B07.
A02–A09 stay ahead of N2.7 in the selected queue.
