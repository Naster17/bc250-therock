# B06.11: relocatable, explicit ORT/MIGraphX inference

Completed2026-10-29 on .104. Source remains ORT
`b81a7866e5a6eff964b9e411535b4c9198651a9f`; existing MIGraphX/ROCm/Torch binaries
and runtime image are unchanged. Package tools checkpoint `bc0f0b0b9`, public
caller-context fix `7870a6ee8`, evidence export `fddb14f61`.

## Use

```bash
bc250-ort --ort-info
bc250-ort /path/to/owned-inference.py
bc250-ort --runtime-profile rccl-single /path/to/owned-inference.py
```

Inside the script, explicitly select the GPU provider and refuse CPU fallback:

```python
import onnxruntime as ort

assert "MIGraphXExecutionProvider" in ort.get_available_providers()
options = ort.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(
    "/path/to/supported-model.onnx", sess_options=options,
    providers=[("MIGraphXExecutionProvider", {"device_id": "0"})],
)
session.disable_fallback()
# outputs = session.run(None, {"input_name": numpy_fp32_array})
```

This is explicit ONNX inference. Plain `bc250-torch`, `bc250-train` and
`bc250-migraphx` retain their existing CPU ORT package. Torch operations are not
automatically routed through this provider, and ONNX inference does not connect
Torch autograd through the model. Tested scope is six static FP32 IR10/opset18
fixtures, not every model/dtype/opset or a performance/training qualification.

The launcher supports ordinary runtime profiles, mounts, `--env`, `--env-file`
and shared-memory options; `--shell` is refused. Caller environment/options cross
elevation using the existing private context validation. Bare `--env NAME` and
inherited `HF_*` variables were tested from actual UID1000. The child selects the
package first on Python's path and forces the admitted low-precision-conversion/
exhaustive-tuning flags off. CPython3.12 runs in the qualified glibc image, not
native musl Python. The GPU provider still requires explicit session selection.

## Package and activation

- Version: `onnxruntime-migraphx 1.31.0+gfx1013.b0611` (archive, not a pip wheel).
- Root: `/opt/bc250-stack/sdks/bc250-ort-migraphx-1.31.0-gfx1013-cp312-b0611`.
- Registration: `/opt/bc250-stack/sdks/ort-migraphx.json`; retained version record
  is the package name plus `.json` in the same directory.
- Wrapper: `/usr/local/bin/bc250-ort` points into versioned
  `/opt/bc250-stack/tools/ort-b0611`.

Metadata is materialized using the fields ORT normally creates at packaging time,
with separate distribution metadata. The previous `build_and_package_info`
warning is gone; tests turn that specific warning into an error, rather than
suppressing it. Native ORT base version remains1.31.0; Python package identity
adds the qualified local version.

Five copied ELF files have RUNPATH normalized to `$ORIGIN`, retaining DT_NEEDED.
Independent local inspection confirms identical `.text` bytes against the B06.10
originals. Original binaries were not modified. Patchelf version/hash and before/
after identities are in the manifest. Two private extracts, including spaces,
run with the original build tree and `/sdk` absent. Registered MIGraphX SDK and
native ROCm remain external dependencies, pinned to the qualified image/runtime
and SDK identities. This is not standalone ROCm or cross-device qualification.

504 regular files/two contained links include matching Python/core/providers,
headers, licenses/notices, original ORT source snapshot plus repair files/pins,
13 original dependency archives with package-path/hash mapping and build settings.
The source checkout remains shallow; no signed-tag/full-history claim. Python
inference is qualified here; the included C++ headers/core are not a blanket new
C API consumer qualification.

Shared daily runtime lock covers identity checking and execution. Installation,
registration changes and the wrapper repair use its exclusive lock. Root path/
mode ownership, pinned index/manifest, complete file hashes/set, contained links
and SDK/runtime cohort identities are checked. Package mounts are read-only.
A mismatched SDK/runtime refuses until separately requalified. Ordinary launcher
home/cwd visibility is retained; only the private relocation tests omit old home/
build trees.

Rollback keeps all data and tools, removing only the opt-in registration. The
public wrapper then refuses. Choose a fresh receipt filename for each operation:

```bash
WS=/home/nik/bc250work/rocm-10.2-gfx1013
doas python3 "$WS/helpers/deploy_onnxruntime_b0611_on_host.py" unregister \
  --receipt "$WS/logs/ort-unregister-manual-1.json"
doas python3 "$WS/helpers/deploy_onnxruntime_b0611_on_host.py" register \
  --receipt "$WS/logs/ort-register-manual-1.json"
```

Live unregister/refusal/re-register, lock exclusion, identity mismatch and
unchanged plain CPU ORT were exercised. Payloads, old images/releases/sources/
builds/caches/checkpoints/path links and boot fallbacks remain.

## Evidence

174 accepted checks:158 guarded checks/eight accepted guards,14 host registration/
recovery/refusal/default checks and two actual normal-user activation checks.
48 GPU graph evaluations and24 actual per-model GPU profiles cover two extracts
and ordinary daily/RCCL activation. Earlier ordinary acceptance also passed90
checks/24 GPU evaluations; all14 guard records acknowledge. No numeric failure,
tolerance relaxation or kernel-log change.

Each main pass has34 checks: retained six graphs/two changed inputs versus CPU
FP64 and retained CPU ORT, shape/type/host-input guards, actual GPU node assignment
with CPU fallback disabled, wrong-input refusals, CPU inference after GPU, exact
Torch HIP backward, loaded origins, package identity and read-only mount. Retained
core7/installed CPU ONNX4/daily5/RCCL6 also pass. No broader allocator/concurrency/
dynamic-shape/autograd/model-quality or speed claim.

The first installed wrapper lost bare caller `--env NAME` after elevation. A
normal UID1000 run reproduced the refusal. It now resolves caller options before
elevation and reuses existing private-context validation; a second normal-user
run proves both bare and inherited variables. Initial wrapper/records are kept
at `WS/recovery/ort-b0611-before-context`; refreshed ordinary acceptance-r2 passes.
Archive/module/kernel bytes were unchanged by this host-wrapper repair.

All3,553 final protected native hashes,35 links/four AOT DBs and136 workstation-
backed retained paths match. Full installed SDK/package identities also pass.
Same r2boot `83f6cb1b-4540-4149-839e-a6cf7cb80b63`. No owned jobs/containers/armed
guards remain. No .105 contact, kernel/BIOS/image/ROCm/Torch/default change. The
historical Python CPU fault remains unresolved and distinct from fixed VM-heap
corruption. No compilation was needed; ccache and previous builds remain intact.

Accepted `.104` paths, with `WS=/home/nik/bc250work/rocm-10.2-gfx1013`:

- Package: `WS/tmp/onnxruntime-b0611/package-r1/final.json`, PID14494 exited0.
- Relocation: `WS/logs/onnxruntime-b0611-relocation-r1/final.json`, PID15155 exited0.
- Installation: `WS/logs/onnxruntime-b0611-install.json`, PID16120 exited0.
- Final ordinary acceptance: `WS/logs/onnxruntime-b0611-acceptance-r2/final.json`,
  controller `WS/logs/onnxruntime-b0611-acceptance-r2-controller.log`, PID19704 exited0.
- Export: `WS/tmp/onnxruntime-b0611/export-r1`, PID21375 exited0.

Local `kernel/artifacts/onnxruntime-b0611` retains archive, evidence, extracted
package/raw reports, ELF/verification/preservation/cleanup receipts. Package
350,607,989B SHA
`cdac3eab84dd0803ae85eac9e46d657ea67bd267db0bbbaf2ac53c2ec21feae3`.
Evidence1,872,893B SHA
`8bffd0ce5f114ff5e08ee51c32f8a3b9bd5919f59644b5448045b4b65fa22269`;
256 raw files8,720,980B. Local29 unit tests, selected critical Ruff, Black,
py_compile/diff checks pass; full upstream CI not run.

Next B06.12: one bounded end-to-end Torch modern exporter → public ORT GPU
qualification with independent CPU/eager-GPU references, then practical B06 gap
review before selecting further expansion. Preserve installed defaults/.105;
this batch did not qualify arbitrary Torch exports or promise training speedups.
