# PLAN2-01 — production-readiness stability soak

Completed 2026-11-14 on the .104 release (r2 kernel `7.1.5-bc250rocm-r2`,
release `10.2-gfx1013-20261111-138a66f52`). First item of
[PLAN2.md](../../../PLAN2.md). **Result: PASSED** — a sustained multi-hour LoRA
run with exact checkpoint/resume is reliable on the stabilized host at stock
clocks.

## Headline

**20 chained bounded segments · 200 optimizer updates · 2.073 h accumulated
training · all 9 reliability checks green.** Global steps contiguous **1→200**
(exact resume across every segment boundary), no crash/drift/leak, kernel log
clean (modulo pre-declared benign governor noise), no foreign containers, boot
ID unchanged throughout.

| Metric | Value | Bound | Verdict |
| --- | ---: | --- | --- |
| Accumulated training | 7461.7 s (2.073 h) | ≥ 7200 s | pass |
| Updates / segments | 200 / 20 | — | — |
| Step continuity | 1→200 contiguous | exact resume | pass |
| Per-update drift (tail/head median) | 0.9992 | [0.8, 1.3] | pass |
| Cross-segment memory ratio | 1.000 | ≤ 1.15 | pass |
| Temperature (inter-segment max) | 65.85 °C | ≤ 90 °C | pass |
| Mean update time | 37.31 s | — | — |
| Finite + non-skipped updates | 200/200 | all | pass |
| Loss | 2.936 → 2.025 (finite, decreasing) | finite | pass |
| Adapter changed each segment | 20/20 | all | pass |
| Guardians passed | 20/20 | all | pass |

Post-run idle temperature 70.85 °C (residual heat; still well within the 90 °C
envelope). VRAM peak ~6.17 GiB throughout (no growth).

## Method

- **Chained bounded segments.** The qualified guardian envelope is 460 s child /
  600 s boot-guard, so a multi-hour run is accumulated as many ≤460 s segments
  joined by **exact checkpoint/resume** (this also validates restart reliability).
  Each segment trains 10 more optimizer steps (cumulative `max_steps`), saves a
  checkpoint, and the next resumes it. Shared output dir per segment; large
  `num_train_epochs` so the app's epoch-based final-save suppression never fires.
- **App.** The authorized lora-smoke `train_two_host.py` (Transformers 5.22.1),
  single-board, Qwen3.5-2B rank-128/alpha-256 all-linear LoRA, FP16, eager
  attention, gradient checkpointing, global batch 4, seed 42, constant LR. Run
  **without** `BC250_LORA_BENCHMARK` (benchmark mode refuses resume); the app's
  `Audit` callback still records per-update finite/grad/skip/seconds/peak rows
  and asserts the adapter changed. Owned D67 synthetic data (`owned-1024.jsonl`,
  hash-verified); no user data.
- **Guardians.** D71-hardened (`guard_soak.py`): SH runtime lock, nonce-matched
  600 s boot-guard ack, 460 s bounded child, `timeout 430` podman backstop,
  kernel-log/registry identity, foreign-container refusal, 900 s orchestrator
  session. Durable host state (`soak-state.json`) makes the run resumable after
  interruption.
- **One pre-declared refinement** (`kernel_noise.py`): the kernel-log identity
  check ignores ONLY the benign `amdgpu …: Unsupported clock type` governor
  clock-query warning; any other dmesg change still fails. Tests prove real GPU
  faults (ring timeout / GPU reset / VM fault / panic) are NOT filtered. The
  original D71 guardian is unchanged; only the soak copy uses the filter.
- **Clocks.** Stock 2000 MHz under load (the governor's qualified safe-point
  range). No overclock during the successful run.

## The incident journey (why this took several attempts)

1. **seg2 spontaneous host reboot** (~00:15 UTC, first attempt). Read-only root
   cause: a persistent **3500 MHz SMU overclock** (`bc250-smu-oc`, applied at
   boot) + **`amdgpu.gpu_recovery=0`** → GPU hang under load → hardware/firmware
   reset. Ruled out boot-guard (deadline not reached), watchdog (SP5100 TCO
   `bootstatus=0`), kernel panic (no `panic=`). **Host-level, not a stack
   defect.** See `ROOT-CAUSE-analysis.md` + `reboot-investigation/` +
   `INCIDENT-seg2-host-reboot.md` + `seg2-crash/` in the evidence archive.
2. **User stabilized the host** (stock 2000 MHz under load).
3. **Benign kernel noise.** The stabilized governor logs `Unsupported clock
   type` warnings; the strict kernel-log invariant flagged them even though
   training was perfect. → pre-declared `kernel_noise.py` refinement (above).
4. **`amdgpu_top` held the GPU** as non-root → the d09 preflight correctly
   refused. User closed it (or runs it via `doas`).
5. **Clean 2 h run.** seg2→seg19 all green; 200 updates; all checks pass.

## Boundaries

- Single-board (.104) LoRA soak; owned synthetic data only; no user application;
  no default/package/release change; installed modules unchanged. The host
  governor/clock/watchdog/kernel were **not** modified by this work (read-only
  investigation only; the user stabilized the host themselves).
- The reliability gate is met **at stock clocks on a stable host**. An aggressive
  SMU overclock + `gpu_recovery=0` reproduces spontaneous reboots and is outside
  the qualified envelope.
- This validates sustained operation + exact restart reliability over 2 h; it is
  not a multi-day soak nor a universal model/hardware guarantee.

## Verification

- Workstation: 27 soak unit tests (summarizer bounds, app/guardian config,
  kernel-noise filter incl. real-fault non-filtering, guardian refinement,
  hardened backstop); full bc250 suite **377 pass**, Black clean.
- On-board: 20 guardians passed; boot `2b5a9904…` unchanged across the run;
  kernel log clean (benign governor noise filtered, pre-declared); registry
  unchanged; 0 foreign containers; all boot-guards acknowledged; final state 0
  containers, r2 kernel intact. .105 untouched.

## Conclusion

The BC-250 stack is **reliable for sustained multi-hour LoRA fine-tuning with
exact checkpoint/resume** at stock clocks: 200 updates over 2.07 h with no crash,
drift, memory growth, or numerical fault. PLAN2-01's production-readiness gate is
**met**, conditional on a stable host configuration (stock clocks; the user's
3500 MHz overclock experiment is the identified instability source and is
outside the qualified envelope). Next PLAN2 items (NF4/QLoRA, native sparse/eig,
…) remain optional and await user direction.
