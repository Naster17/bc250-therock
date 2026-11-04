# B07.12: coherent FP32/FP16 CK and repaired-MIOpen artifact

Completed2026-11-04 on .104. Tools `c64debffb`; library sources `96e5a7dfc43`,
Torch `a524980` and installed native libraries `8389197c1` unchanged. This batch
packages the B07.10 FP16/FP32 facade and B07.11 resource-repaired MIOpen. No
native rebuild, installation, registration or ordinary default change.

The artifact supplies packed2D FP32/FP16 forward, exact alpha1/beta0, restricted
descriptors/pointers/workspace, six FP32 and nine FP16 configurations. The bundled
HIP-direct gfx1013 forward retains FP32 accumulation. Native backward remains;
BF16, mixed types, casts, transpose/fused groups and unsupported shapes refuse.
The installed `bc250-ck` remains the older FP32 artifact until B07.13.

The package keeps MIOpen in `runtime/` and the CK plugin in `lib/`. A fresh worker
uses private MIOpen ahead of the existing runtime SDK, verifies27 actual mapped
MIOpen/HIP/C++/ROCm library identities, native gfx1013:xnack- and API5/66 required
exports, then runs the caller script in that verified process. The outer parent
stays alive; there is no exec replacement after HIP initialization. Loader paths
do not expose the plugin when its explicit CK setting is absent.

Only copied ELF files are normalized: MIOpen RUNPATH `$ORIGIN`, plugin/probes
`$ORIGIN/../runtime`, with DT_NEEDED unchanged. Eight ELF files are audited.
The archive retains source/config/license/provenance snapshots, exact original
resource-relink recipes and source/archive pins. It depends on the matching
runtime SDK; it is not a standalone or cross-version ROCm distribution.

## Qualification

Two independent extracted roots, including spaces, each run on daily and
RCCL-single with original source/build/candidate paths unmounted. Tests check:

- Every payload hash, actual private MIOpen mapping and27 native identity pins.
- Native pin/architecture/payload/path/extra-file/symlink refusals and exact reasons.
- Genuine FP32 two-key plus FP16 six-key FindDb/PerfDb examples per profile,
  nonexistent-only seeding, no replay mutation and no implicit CK activation.
- Public solver calls, six FP32/nine FP16 per-ID invokers, mixed-dtype same-handle
  references, pointer/descriptor/scratch refusals and Torch first gradients.
- Both public FP32/FP16 tuning-record replays using independent CPU double
  references; caller argv/importdir/env/cwd/native-backward behavior.
- Retained installed FP32 option, core, CPU ONNX and daily/RCCL consumers.

**78 top-level checks/10 final ACK guards**, with48 nested reference checks,
pass.1496 FP16+1176 FP32 native convolution references in relocated artifacts,
plus586 installed FP32 references;24 Torch convolution output/all-first-gradient
fixtures.24 entry identity refusals/four existing-seed refusals,128 half+128 float
absent-path refusals,736 half API/1908 runtime and488 float API/984 runtime
refusals.29 local tests/Black/critical Ruff/Python syntax/diff checks pass.
Independent local archive hashing and ELF inspection verify39 files/eight ELFs.


Bounds and guards are unchanged from B07.9–11. This qualification adds relocation
and activation evidence, not universal arithmetic/backward/graph/concurrency or
performance coverage. B07.11's one-of-six1.17–1.18x fixture gain remains a narrow
historical measurement; this package forces no global solver choice.

Three packaging attempts are preserved. R1 assumes the final source FindDb still
has only six FP16 keys, but later B07.11 gradient tests added entries. R3 selects
the six authoritative native-search shape keys and copies their existing lines
verbatim; it does not invent performance records. Reference/gradient checks use a
separate writable DB so the seeded replay DB remains unchanged. R2 finds a CPU
packaging metadata lookup error before ELF work; R3 uses the actual daily image.
The first relocation gate passes identity then refuses cross-bind-mount fixture
hardlinks. The controller now creates fixtures on the host filesystem, sharing
only immutable payload inodes and copying manifests independently. Accepted r2
runs the same package-r3; no binary/reference/tolerance changes for this fix.

3965 protected hashes/36 links/four AOT databases/136 workstation backing files
unchanged. Same d10514df-543d-433b-a352-0e3ce79bc7a8 boot/r2 kernel throughout this
batch; it differs from B07.11's older boot. Kernel logs unchanged through cleanup,
no KFD recurrence, no owned running jobs/containers/armed guards;11 total guards
ACK. User explicitly authorizes stopping container6fcfde0398e9 (`lucid_wiles`);
it is stopped and preserved, not removed. No .105/user-app access or deletion.
Free12,879,323,136→10,253,754,368B. PackagePIDs8781/9525exit1,
10202exit0; qual12590exit1/13470exit0; export17389exit0.


The historical Python CPU fault and B07.7 KFD-return uncertainty remain separate
from fixed VM-heap corruption. Fresh workers pass the bounded kernel-log gate;
this is not a kernel root-cause repair claim. Ordinary native rollout of the
FP16 precision repair remains pending compatibility qualification.

## Artifact and use

Local `kernel/artifacts/ck-plugin-b0712/bc250-miopen-dl-gfx1013-b0712.tar.gz`:
66,651,729B SHA256 `a0d47ed4ba4df354cc7ea898a285eefa051f49693667386567ea392f66ddd91e`.
39 files/505,424,918B extracted. Host release is at
`WS/tmp/ck-plugin-b0712/package-r3/bc250-miopen-dl-gfx1013-b0712.tar.gz`.


After extracting and assigning the directory to `BC250_CK_DIR`:

```sh
bc250-torch --mount-ro "$BC250_CK_DIR" "$BC250_CK_DIR/bin/bc250-ck.py" check
bc250-torch --mount-ro "$BC250_CK_DIR" "$BC250_CK_DIR/bin/bc250-ck.py" run /absolute/app.py
```

For independent writable example seeding, use the entry's `seed-db --cohort daily`
(or `rccl`) `--destination NEW_DIRECTORY`, then set MIOPEN_USER_DB_PATH and
MIOPEN_CUSTOM_CACHE_DIR explicitly. Existing destinations refuse. The installed
FP32 command is independent; the archive is not selected by normal launches.
See the packaged README for scope and runtime requirements.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`; accepted commands:

```sh
python3 helpers/package_ck_plugin_b0712_on_host.py --attempt r3
python3 helpers/qualify_ck_plugin_b0712_on_host.py --attempt r2 --package-attempt r3
python3 helpers/export_ck_plugin_b0712_on_host.py --attempt r2 --package-attempt r3
```

Populate fresh resources from the recorded qualified source before packaging;
existing destinations are preserved. CPU-only packaging uses twoCPU/2GiB for the
ELF/identity containers, no build network/GPU/cache reset. GPU phases use SH
runtime lock, existing preflights and600s watchdog gates. No EX prefix update.

`WS/logs/ck-plugin-b0712-qualification-r2/final.json`, matching r2-controller.log,
`WS/tmp/ck-plugin-b0712/{package-r1,r2,r3,r3-extract-a,r3-extract with spaces,
export-r2}` preserve reports/partials/extracts. Local evidence.tar.gz:
7,079,195B SHA256 `adb8ec1f0f84ec3bddc122c25ccae89969b982a6c6c5123b794b1c765c13a650`;
379 raw files/20,168,794B. All staged helpers and payload hashes
match the accepted revisions; the release archive is retained separately.


Next B07.13: versioned explicit .104 installation/registration and updated
`bc250-ck` activation for this coherent artifact, exact identity/refusal/live
rollback and ordinary daily/RCCL caller-context tests under the EX update lock.
Retain old FP32 artifact/tool/version records and ordinary native defaults/.105.
