# B06.8: registered opt-in MIGraphX SDK on .104

Completed 2026-10-28. SDK source remains
`4797a61125bbec318005969f8497dbd50849cd6e`; implementation checkpoint
`de93d4fe992383ef4787d494706b52d5c7f707b6`, evidence export
`5f5a73b7a` (full revision in workspace manifest). No native rebuild or compiler
cache reset was needed: installation uses the previously qualified B06.7 archive.

## Use

On .104:

```bash
bc250-migraphx --sdk-info
bc250-migraphx /path/to/owned-inference.py
bc250-migraphx --runtime-profile rccl-single /path/to/owned-inference.py
```

Inside the script, `import migraphx` exposes the installed SDK. Parse supported
ONNX with `migraphx.parse_onnx`, compile with
`program.compile(migraphx.get_target("gpu"), offload_copy=True, fast_math=False)`,
then call `program.run` with named NumPy FP32 inputs. Shape/operator/dtype
qualification still applies. `--env`, `--env-file`, `--mount`, `--mount-ro`,
`--shm-size` and ordinary Python arguments are forwarded through `bc250-torch`.
External script parents and paths containing spaces are supported; `--shell` is
refused. CPython 3.12 runs inside the existing qualified glibc image, not native
Alpine Python.

The new host launcher delegates to the existing ordinary `bc250-torch` launcher.
It holds the shared daily runtime lock across SDK verification and execution;
registration changes require the same exclusive lock. It checks root-controlled
paths, the pinned manifest/index, every SDK file hash, the exact file set,
contained symlink identities, selected runtime manifest and image. SDK is mounted
read-only. Existing home/cwd visibility follows ordinary launcher behavior.
Startup verification reads the complete SDK; no startup-latency claim was made.

Plain `bc250-torch`/`bc250-train` retain their defaults and do not import MIGraphX.
This explicit inference API does not automatically replace Torch operations,
provide MIGraphX autograd or register an ONNX Runtime GPU execution provider.
ONNX Runtime's CPU provider remains installed and qualified.

## Installation and rollback

| Item | .104 path |
| --- | --- |
| SDK | `/opt/bc250-stack/sdks/bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067` |
| Active registration | `/opt/bc250-stack/sdks/migraphx.json` |
| Retained version record | `/opt/bc250-stack/sdks/bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067.json` |
| Host launcher | `/usr/local/bin/bc250-migraphx` |
| Versioned launcher/modules | `/opt/bc250-stack/tools/migraphx-b068/` |

The archive SHA256 remains
`11f7f7ab1923132131dc0118af62fbd425e55cfd802d86fdcb57e4952583c8ba`.
Installed SDK has 566 regular files (including its index), 12 contained links.
Installer requires 4 GiB ordinary free space and refuses existing destinations.
Root ownership/modes are normalized; library bytes and old sources/builds/caches
remain intact. Empty staging directory and deliberately invalid refusal fixtures
are retained as owned evidence. No old artifacts were deleted.

Rollback unregisters only the opt-in registration. SDK, version record, launcher,
source archive, runtime/boot fallbacks and earlier test evidence remain. The
launcher then refuses activation. Use a fresh receipt filename each time:

```bash
WS=/home/nik/bc250work/rocm-10.2-gfx1013
doas python3 "$WS/helpers/deploy_migraphx_b068_on_host.py" unregister \
  --receipt "$WS/logs/migraphx-b068-unregister-manual-1.json"
doas python3 "$WS/helpers/deploy_migraphx_b068_on_host.py" register \
  --receipt "$WS/logs/migraphx-b068-register-manual-1.json"
```

These commands run preflight and obtain the exclusive lock. Re-registration
verifies the retained SDK/tools; runtime drift still refuses subsequent launch
until separately requalified. Unregister/register was exercised on the live
opt-in registration, without changing a runtime manifest. Runtime-drift and
SDK-byte/extra-file refusal tests used root-owned scratch copies, not active
runtime/library modifications.

## Evidence and scope

Accepted `.104` command:

```bash
python3 helpers/qualify_migraphx_b068_on_host.py --attempt r1
```

Controller PID8265 exited0; log
`WS/logs/migraphx-b068-qualification-r1-controller.log`, final receipt
`WS/logs/migraphx-b068-qualification-r1/final.json`. Export PID10260 exited0;
`WS/tmp/migraphx-b068/export-r1` contains archive/verification metadata.
No owned build/test/export jobs or containers remain; ten guards acknowledged.

| Gate | Checks |
| --- | ---: |
| Daily installed activation / Python / public C API and CLI | 8 / 50 / 6 |
| RCCL installed activation / Python / public C API and CLI | 8 / 50 / 6 |
| Retained core / CPU ONNX / daily / RCCL consumers | 7 / 4 / 5 / 6 |
| Host registration, rollback, identity/refusal, lock exclusion and default absence | 14 |
| Actual ordinary UID1000 elevation and SDK info identity | 1 |
| Total | 165 |

The two Python import orders are MIGraphX-first and Torch-first. Six retained
FP32 ONNX IR10/opset18 fixtures cover pointwise/broadcast, reduction, dense,
transpose/MatMul, convolution and serialized initializers; changed inputs,
serialization, explicit HIP copies and CPU-double/CPU ORT references retain
B06.7 tolerances. 68 GPU graph evaluations total. Installed HIPRTC helper processes
and native SQLite/ROCm library origins are observed. Activation checks also
exercise external script sidecars, spaces, read-only SDK mount, explicit env,
selected NCCL cohort, SDMA0 and exact Torch HIP first backward.

All 1,681 native protected hashes and 29 recorded links/four AOT DBs match.
All 136 workstation-backed retained paths match before and after qualification.
Full installed SDK identity is separately checked, including during export.
Kernel logs match on r2 boot `83f6cb1b-4540-4149-839e-a6cf7cb80b63` throughout
this batch; this is a new pre-task boot, not continuity with B06.7's boot.
No agent reboot, active ROCm/Torch/image/default change, user application work or
.105 contact occurred. Export ordinary free space: 41,137,750,016 bytes.

Local `kernel/artifacts/migraphx-b068/` contains evidence archive, extracted raw
logs/tools/guards, registration, verification and backing preservation receipts.
Archive: 623,640 bytes, SHA256
`c0e0ce195e4507d57694422dcc0ffd878eaeb686f6aa5627e45cd683cf991396`;
188 raw files / 2,698,354 bytes. Local 29 stack unit tests, Black, py_compile and
diff checks pass; Ruff unavailable and full upstream CI not run.

No arithmetic failure or tolerance relaxation was needed. No model performance,
all-model/opset/dtype, MIGraphX graph backward, ORT GPU provider or cross-version
compiled-program portability qualification. Historical unknown Python CPU fault
remains unresolved and distinct from already-fixed VM-heap corruption.

Next B06.9: audit pinned ONNX Runtime MIGraphX EP/API/dependency and storage
requirements, preserving installed CPU ORT and this opt-in SDK. Any build must be
an isolated candidate; no provider/default switch before GPU reference and
retained Torch/daily/RCCL checks. Broader model/dtype/performance admissions stay
separate.
