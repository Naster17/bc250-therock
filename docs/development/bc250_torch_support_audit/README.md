# Installed Torch support audit evidence — 2026-10-16

See the [support matrix and stable work-item IDs](../bc250_torch_support_matrix.md).
Source helper: `build_tools/bc250/audit_torch_support.py`. This batch changes audit
code/docs only; no Torch/ROCm/kernel rebuild or production dispatch change.

`merged-results.json` selects the final valid probe for each of51 unique targeted
configurations.35 pass,6 reproduce MIOpen defects,10 are absent/configured-off or
refused API paths. These are mixed correctness/gradient/contract/availability
checks, not a percentage of supported Torch.35 passes include a finite AMP update,
RNG replay and graph replay; their stronger prior numerical gates are retained.
An audit collector exit0 means collection completed; inspect individual statuses.

- `torch-support-audit`: original45 cases; config/ELF/libraries/dispatcher dump.
- `torch-support-refined`: valid INT8 dimensions, direct GPU backend/head-width
  probes with CPU math references, group1 convolution, optional package discovery.
- `torch-support-conv-isolation-r2`: inference and requested-gradient configurations,
  scoped GEMM-off diagnostics, corrected AMP dtype check. GEMM-off does not force
  a single Direct solver and is not enabled in production.
- `torch-support-conv-isolation`: retained helper syntax failure before GPU use,
  caused by a duplicate env keyword during editing. Exit1, guard ACKed, no dmesg
  change. Corrected locally with py_compile before rerunning under a new label.

Probe corrections are not backend fixes: initial INT8 M=16 violated Torch's
M>16 contract; final32x32x32 reaches the unsupported Lt heuristic. Initial tiled
reference incorrectly called the GPU-only helper on CPU/FP64; corrected probe
calls it on native GPU and gets the genuine head256 refusal. AMP originally
asserted FP16 loss; autocast correctly promotes square/loss to FP32 while matmul
is FP16. Final AMP update passes at init_scale128. No numerical tolerance changed.
Original/refined helper copies remain locally in ignored
`kernel/artifacts/torch-support-audit/`; complete prior logs stay on .104.

Convolution distinction: the initial combined group1 trainable-input/trainable-
weight fixture compared forward successfully then failed backward. New input-
only/weight-only requires-grad variants failed in forward before a directional
backward could run. They therefore do not prove which backward kernel is broken.
Inference-only forward passed. All use B2/C4/L19,W6x4x3,stride2,pad2,dilation2.
Workspace480/provided0 messages and elapsed<=0 are both retained. Disabling GEMM
removes the workspace warning, but invalid timing/no suitable algorithm persists.
No guessed root cause or workaround is declared qualified.

LSTM is a separate newly reproduced defect: B2/L5/input3/hidden4, CPU-float64 forward
comparison passed; GPU backward failed compiling MIOpen's legacy-CK reduction,
`gridwise_generic_reduction_first_call_threadwise_reduce_partial_dims.cpp`.
Compiler diagnostic: `No CK_AMD_GPU_GFX* macro defined`. This is not the missing
optional MIOpen CK grouped-conv DSO and not the already-patched standalone CK Tile.

Key source locations to inspect next:

- Torch `aten/src/ATen/native/miopen/Conv_miopen.cpp` and MIOpen
  `src/conv/solver_finders.cpp:535`, `src/solver/conv/gemm.cpp` workspace checks;
  isolate size-query/invocation and profiling/handle/stream behavior using C API.
- MIOpen `src/reducetensor.cpp`, `src/include/miopen/solver/legacy_ck_common.hpp`,
  `src/legacy_composable_kernel/composable_kernel/include/utility/config.hpp`.
- MIOpen `src/solver/ck_impl_lib_loader.cpp`, `src/ck_impl/CMakeLists.txt`,
  top-level database installation and `src/find_db.cpp`/`src/db_path.cpp.in`.
- Torch `cmake/public/LoadHIP.cmake`, `cmake/Dependencies.cmake:1124`,
  `aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.cpp`; installed hipsparselt CMake
  exports exist but the existing Torch configure cached hipsparselt_DIR-NOTFOUND.
- Torch `aten/src/ATen/native/cuda/Blas.cpp:851`,
  `aten/src/ATen/cuda/CUDABlas.cpp:2319` show dense INT8 routing through hipBLASLt.

The installed database directory has only lgbm_model_meta.json/lgbm_rank.bin/
lgbm_pcfg.bin. No gfx1013 system db/kdb is installed. User ufdb/ukdb caches exist;
`host-inspection.json` records counts/filenames without reading their entries.
The `gfx1013_20` key matches HIP20 WGP multiprocessors on the40-CU board; do not
rename databases or force a different architecture. All19 existing DSOs out of21
examined loaded, including HIP/compiler/BLAS/DNN/FFT/sparse/solver/RNG libraries and
optional hipTensor/hipDNN/hipSPARSELt. RCCL and gfx1013 CK grouped-conv plugin absent.
Optional Python package presence refers to the reviewed stable container runtime,
not every host venv. Qwen3.5's separate Transformers5.19.0 project overlay remains
as previously deployed; stable consumer overlay here is5.2.0.

All runs used the existing exclusive runtime lock and pinned immutable image,
native gfx1013, SDMA0 and heap-off defaults. The host runner armed600s guards and
retained command/kernel-before/kernel-after/exit/guard logs; guards ACKed, kernel
logs byte-identical including the helper failure. First batch core limit256MiB
was diagnostic only; no crash/core occurred and no global core policy changed.
Current libtorch_hip SHA578a3583e78ad0009779d77bd65591fcf8119dec441ea81dde7be56d9b9c11d5
and boot a287fd16-c97b-4d88-8055-b0553a2f7631 retained. Existing source/build/cache/
checkpoint/release/compatibility links preserved; normal JIT cache additions are
expected. No package installation, native build, runtime/prefix/boot/BIOS/governor/
.105/user src/ change. No agent build/test/container/recovery deadline at handoff.

Replay with root on .104 under the lock, never nest bc250-torch's shared lock:

```sh
WS=/home/nik/bc250work/rocm-10.2-gfx1013
python3 "$WS/helpers/update_dev_prefix.py" -- \
  python3 "$WS/helpers/validate_tiled_sdpa_on_host.py" \
  --label NEW_UNIQUE_LABEL --script audit_torch_support.py
```

Use `--test-arg=--only --test-arg=CASE` to select a bounded subset. Run output
contains expected failures; require independent references for any proposed fix.
Next A01: direct-MIOpen minimal reproduction and workspace/profiling isolation;
A02 legacy reduction is separate. N2.7 remains queued per the user's new request.
