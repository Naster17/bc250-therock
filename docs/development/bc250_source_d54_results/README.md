# PLAN-07 / D54: isolated clean-build admission and retained closure

Completed 2026-11-11 on .104. Implementation 742385f4f.
[Receipts](artifacts.json), [replay summary](verified-results.json).
**PLAN-07 remains OPEN**: this is a bounded admission probe, not a full Torch
rebuild, wheel, or license/publication clearance. .105 was not accessed.
The clean probes ran 2026-11-10/11 in the prior session; the retained audits,
RCCL byte-comparison, evidence transfer and closeout completed in this pass.

## Method

`run_clean_probe_d54_on_host.py` runs as root on bc250 under the board
reservation plus shared runtime lock, with fail-closed preflight (watchdog
active, no unrelated Docker/Podman workload, only the native cyan-skillfish
GPU client), 12 GiB available-memory and 20 GiB disk admission, 3 GiB host /
5 GiB disk running floors, a 720 s deadline, owned-container-only stop, and
scratch removal after every attempt.

The container is the preserved original builder image
`28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06` (D52 OCI
identity), `--rm --pull=never --network=none --read-only`, 4 CPUs (0-3),
10 GiB memory+swap, 512 PIDs, 2 GiB tmpfs `/tmp`. Read-only mounts: retained
ROCm build prefix `build/dist/rocm` (SDK a6dc1c88 build input, distinct from
the active D49 SDK 2a975370), `rccl-b034/install-r1`, ccache data
(`CCACHE_READONLY=1`, `CCACHE_NOSTATS=1`, 20 G config kept, no reset), and the
D52 recipes. The fresh D52 source kit
(SHA `6867ed719c201cf7bc368800aa64c5e16a2e3a4c3c03485ad502684e2feb5723`) is
hash-verified and extracted to scratch per attempt.

Clean phase: the archived `build-fresh-<profile>` recipe environment is
sourced up to the pip command; `PIP_NO_INDEX=1`; `offline_inputs_d54.cmake`
forwards `PYTHON_SIX_SOURCE_DIR`; `BUILD_CUSTOM_STEP` injects the bounded
native step. That step compiles
`aten/src/ATen/native/hip/BatchLinearAlgebraEig.hip.o` (plus
`torch/csrc/distributed/c10d/NCCLUtils.cpp.o` for rccl-single) and the NNPACK
peachpy object `2d-fourier-8x8.py.o` with `ninja -j 4`, requires a gfx1013
offload bundle via `llvm-objdump --offloading`, links and runs a test binary
against amdhip64/MIOpen/hipblas/hipfft/hipsparse/hipsolver/rocprofiler-sdk
(plus rccl for rccl-single), and refuses any `ldd` "not found". It then writes
`target-passed` and exits 54 by design: a targeted build cannot produce the
editable wheel, and the wrapper asserts pip rc==1 with the marker present.
The pip wheel failure at the end of `probe.log` is therefore expected.

Retained phase: read-only b052/b053 trees plus the retained
`pytorch-sparse-followup/third_party` overlay. `audit_link_headers_d54.py`
parses the full `ninja -t deps` database, resolves every `LINK_LIBRARIES`
entry (bare `-l` flags via `g++ -print-file-name`), and checks `ldd`/`readelf`
loader closure for `torch/lib/*.so`, `torch/_C*.so`, the HIP clang, and
librccl (rccl-single) against allowed roots. Retained Ninja graphs reference
the host RCCL path, so the runner passes `RCCL_ROOT` (fixed this pass; the
retained phase had never been executed before).

## Results

| Attempt | Status | Wall | cgroup memory peak |
| --- | --- | --- | --- |
| daily-clean-1 | failed (ccache read-only) | 26.740 s | — |
| daily-clean-2 | agent-stopped during diagnosis | 53.119 s | — |
| daily-clean-3 | agent-stopped during diagnosis | 50.277 s | 195,473,408 B |
| **daily-clean-4** | **passed** | 105.759 s | 891,113,472 B |
| **rccl-single-clean-1** | **passed** | 115.584 s | 1,378,820,096 B |
| **daily-retained-1** | **passed** | 3.806 s | — |
| **rccl-single-retained-1** | **passed** | 3.606 s | — |

Clean probes: offline configure succeeded (daily 34.1 s, rccl-single 31.9 s);
gfx1013 offload bundles present; link/loader closure clean; test binaries
executed. `BatchLinearAlgebraEig.hip.o` SHA
`6c3e6dc0e9731d39b09eeedf4137b2a9960a6a050e7951d1d19e9e078673d56b` is identical
in both profiles. The NNPACK peachpy object differs between profiles (daily
`0a08ef5f…`, rccl-single `cfae2602…`, full hashes in the receipts); both
compiled and are recorded, not equated.

HIP translation comparison against the preserved 20261110 source-audit
variants: **22/22 byte-exact for daily (b052) and 22/22 for rccl-single
(b053)**.

Retained closure:

| Profile | dep records | stale | unique headers | missing headers | link inputs (+bare) | missing link | loader objects | unexpected paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| daily | 3188 | 0 | 17610 | 0 | 55 (+7) | 0 | 11 | 0 |
| rccl-single | 3190 | 0 | 17619 | 0 | 56 (+7) | 0 | 12 | 0 |

Resource minima (probe costs, not performance claims): clean minimum
available memory 13,746,327,552 B and free disk 59,656,708,096 B; retained
minimum available 14,479,544,320 B and free disk 61,650,231,296 B. Daily
clean cgroup events show no OOM/throttling. Wall times above are admission
costs only.

## Hidden dependency found and closed

NNPACK's `DownloadSix.cmake` fetches `six==1.11.0` at configure time unless
`PYTHON_SIX_SOURCE_DIR` is set; the old builds already had it cached, so the
dependency was invisible in retained graphs. Both retained six archives match
NNPACK's pinned checksum. The probe supplies the exact archive offline
(SHA `70e8a77beed4562e7f14fe23a786b54f6296e34344c23bc42f07b15018ff98e9`,
LICENSE SHA `63478682e84e8c98f4c46308995f1f521a6886976e25421827d2a2be03581328`).
The clean offline configure now passes with `--network=none`.

## Verification and hygiene

- 168 local tests OK (`bc250*_test.py`, including two new D54 dependency-parser
  regressions); Black clean; py_compile pass (ruff unavailable on the
  workstation this pass). `git diff --check` clean in both repositories.
- All 30 transferred evidence files SHA-verified against .104; the four final
  evidence sets and the six archive are in `kernel/artifacts/source-d54/`
  (31 receipt entries).
- Host evidence retained at `evidence/source-d54/` (all seven attempts);
  reusable inputs at `tools/source-d54/` (harness, six archive, hip-paths).
  `tmp/` clean; no owned jobs, containers or guards; watchdog active.
- Original sources/builds/caches/prefixes/images/profiles/boot/native controls
  unchanged; ccache read-only, no reset; .105 untouched. No GPU workload ran.

## Next

PLAN-07 continues: full clean-rebuild resource/storage planning from the now
admitted layout, generated-file/license closure, and the four important
packages (PLAN07-10). Historical Python CPU/KFD/.105 clocksource causes remain
unknown and separate from this build-input work.