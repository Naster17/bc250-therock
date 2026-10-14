# BC-250 r2 kernel: preliminary read-only review, 2026-10-14

This is a source/provenance review, not full kernel qualification. No kernel,
module, config, boot, firmware or runtime changes, builds, GPU tests or reboots.
Both hosts were inspected read-only. The N1.8 runtime remains unchanged.

## Identity and reconstructed change surface

Both .104 and .105 run `7.1.5-bc250rocm-r2 #2`, built September16. Both recorded
boot files hash to `792f904835f27574b680d2fd5cfd7e42b348fb3a3d9b39d8e8c87348873c3b0d`.
Their installed `amdgpu.ko.gz` files both hash to
`88e9e465a0fb668075e10c83ccabf473d2482095cf7f27c7ee06b77f24cd2799`;
vermagic matches r2. This does not prove reproducible compilation or loaded
initramfs/module byte identity. Module srcversion was unavailable.

.105 retains the original build tree, `linux-7.1.tar.xz`, `patch-7.1.5.xz` and
build log at `/home/nik/bc250work/alpine-rocm-7.1/`. Local kernel/source is the
already-patched import `6f57bb40d26d1cbec862807fc89a9c61af1b9a39`; do not apply
legacy patches onto it again. Recovered build-tree config and TTM source hashes
still match the recorded .105 provenance anchors.

For this audit, copied the two archives locally and reconstructed only
`drivers/gpu/drm/amd/`, applying36 stable-patch sections in an isolated temporary
baseline. Of2769 archive files,2762 match the imported tree. Seven active files
differ:136 added/3 removed lines. Five additional tracked `.orig` files match the
corresponding baseline sources and are inactive backups. The active delta covers
CU masks, PASID KIQ selection, the TTM null-page guard and four runlist-flush call
sites/helper. No broad AMD driver rewrite was found in this comparison.

The archives' upstream cryptographic authenticity was **not** checked. Comparison
outside the AMD subtree and a full build/module/initramfs audit remain undone.
[JSON evidence](bc250_kernel_audit_20261014.json) records hashes and file deltas;
[active source diff](bc250_kernel_audit_20261014.patch) omits the `.orig` backups.
The diff is evidence against the archive-derived baseline, not an unapplied series
for the existing source tree.

## Findings that need review

1. `kfd_device_queue_manager.c:3806` returns runlist rebuild failures, but all four
   added map/unmap/SVM call sites discard those returns. A failed workaround can
   leave API success unrelated to successful TLB handling. Error/rollback,
   lifetime, lock ordering and completion semantics need review; no such failure
   was injected or reproduced here. Do not blindly return errors after partial
   map/unmap state changes.
2. `gfx_v10_0.c:10143` writes `RLC_PG_ALWAYS_ON_WGP_MASK`; diagnostic probe modes1/4
   restore CC/SPI registers but never restore that RLC register. Mode1 also loops
   over every SE/SH although its parameter description says probe-SE0SH0. These
   diagnostic modes are not read-only probes. Current boot uses mode3, so this
   does not identify a cause of current ROCm failures.
3. Retained packaging patch0005 for eight-core SMU telemetry is absent from this
   imported source: `cyan_skillfish_ppt.c` matches the reconstructed baseline.
   Its GFXCLK path reads `metrics->Current.GfxclkFrequency`, without that patch's
   live-clock query/layout changes. Possible relevance to conflicting clock
   reports is a hypothesis, not an established root cause or instruction to apply
   the retained r1 patch blindly.
4. The retained .104 boot log already contains two Display Core warnings in
   `dal_irq_service_ack`/`dal_irq_service_set` during `amdgpu_dm_hpd_init`, around
   8.93s. They precede N1.8 workload execution. Identical before/after logs mean
   **no new messages**, not a warning-free kernel. Current taint12864 includes
   user-requested taint, WARN, out-of-tree and unsigned-module flags; it does not
   by itself identify the responsible module or a corruption cause.

The proven CLR VM-heap/CSR corruption/default-pool mitigation and unresolved Python
CPU execution fault at0x3b3e6 remain distinct. Nothing here proves a kernel cause
for that Python fault. SDMA/reset/fault migration remain unqualified as before.

## Recommended next kernel action

A clean, separately versioned upstream build with a small reviewed BC-250 series
is justified as an engineering direction. First verify the official source/tag
and full-tree delta; split and review each required change, starting with runlist
failure/ordering and diagnostic register restoration. Review telemetry separately.
Use [kernel.org's release verification procedure](https://kernel.org/signature.html).

Keep the current r2 image/modules/config/boot entry intact. A candidate should get
its own sources, output directory, modules and one-shot boot entry. Acceptance
must include SSH recovery, allocation/free/remap data checks, queue/process
lifetimes and bounded Torch reference checks. New version numbers alone do not
establish better TLB behavior, SDMA/reset support, correctness or performance.

No candidate version was selected or installed in this audit. The next library
roadmap item remains N2.1; this user-requested kernel review is a separate cursor.
