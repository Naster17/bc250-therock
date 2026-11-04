# BC-250 FP32/FP16 DL convolution artifact

This versioned optional artifact contains a matching MIOpen library and CK plugin
for the qualified native gfx1013:xnack- runtime. FP32 and FP16 packed2D forward
require alpha1/beta0 and the recorded descriptor/pointer/workspace contracts.
The bundled MIOpen retains FP32 accumulation in gfx1013 HIP-direct forward.
Plugin backward is unsupported; native backward remains in use. BF16, mixed types,
casts, transpose, fused operations and unsupported convolution configurations
refuse. This is not a standalone or cross-version ROCm SDK.

Inside the qualified runtime, with the extracted directory mounted read-only:

```sh
python /absolute/artifact/bin/bc250-ck.py check
python /absolute/artifact/bin/bc250-ck.py seed-db --cohort daily --destination /absolute/new-db
MIOPEN_USER_DB_PATH=/absolute/new-db python /absolute/artifact/bin/bc250-ck.py run /absolute/app.py arg1
```

From the host, use `bc250-torch --mount-ro /absolute/artifact` before the entry
script argument. Existing runtime locks and launcher mounts/environment apply.
The archive is not registered by these commands; installed `bc250-ck` remains the
older FP32 option until a separately qualified deployment.

The entry verifies all payload bytes before starting a fresh Python worker with
its private MIOpen ahead of the runtime SDK. It checks actual mapped MIOpen/HIP/
C++ dependency hashes, native architecture and API5/required exports before
running the script with its argv/import directory. The parent remains alive;
the worker runs the script through runpy after HIP verification. This avoids
exec-replacing an initialized HIP/KFD process. It does not prove a kernel-level
fix for the earlier B07.7 KFD-return uncertainty.

All libraries come from the existing qualified runtime except this artifact's
MIOpen/plugin. Copied ELF RUNPATHs use $ORIGIN (plugin/probes: $ORIGIN/../runtime), without
source/build-tree directories. External runtime SDK libraries and its sysdeps
remain required and pinned. No kernel compilation, downloads or builds occur at
package activation. Normal JIT convolution compilation can still populate the
caller's writable MIOpen kernel cache.

Daily/RCCL examples contain separate genuine FP32 two-key and FP16 six-key tuning
records. Seed copies require a new nonexistent directory and never merge into
application caches. Use a dedicated writable MIOPEN_USER_DB_PATH and
MIOPEN_CUSTOM_CACHE_DIR. Records are examples, not universal tuning defaults.

B07.11 measured one of six FP16 fixtures 1.17–1.18x faster with tuned CK; one tied
and four favored repaired HIP-direct. This package does not force a global solver
or promise model/training speedups. Other devices, exceptional floating values,
arbitrary concurrency/graph capture, and universal forward/backward coverage are
unqualified. Sources/licenses/provenance and the exact resource-relink recipes
are included; native host objects were reused, not rebuilt from scratch.
