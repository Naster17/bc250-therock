# BC-250 MIOpen DL explicit artifact

This opt-in artifact requires the matching native gfx1013 runtime. It is relocatable
within that runtime; it is not a standalone ROCm SDK or cross-version C++ plugin.
It supplies FP32 packed2D forward only, exact alpha1/beta0; unsupported groups refuse.
Backward, low precision, arbitrary concurrency and general speedups are unqualified.

Inside the qualified `bc250-torch` container, run:

```sh
python /absolute/artifact/bin/bc250-ck.py check
python /absolute/artifact/bin/bc250-ck.py seed-db --cohort daily --destination /absolute/new-db
MIOPEN_USER_DB_PATH=/absolute/new-db python /absolute/artifact/bin/bc250-ck.py run /absolute/app.py
```

Mount the artifact read-only with `bc250-torch --mount-ro /absolute/artifact` before
its Python script argument. Existing launcher runtime locks remain in force.
`run` verifies payload hashes, eight actual mapped runtime libraries, native gfx1013,
and API5/66 exports before enabling MIOPEN_CK_LIB_PATH for that process only.
The script runs as __main__ in the verified Python process with its argv and import
path set to the script directory; this preserves the initialized HIP/KFD context
instead of exec-replacing it. The original candidate directories are not required. No installation/registration,
production whitelist or default change is performed. `seed-db` requires a new
nonexistent directory and never merges into application caches. Daily/RCCL examples
are genuine B07.6 two-key records, not global tuning. Use a separate writable user
DB/kernel cache for further native tuning. HIP-direct remains faster on four of six
B07.5 fixtures; selected public DL wins1.085–1.135x on two recorded moderate fixtures.
No automatic Torch selection or model training speedup is established.

`manifest.json` pins all payloads and exact ABI-library hashes. Its integrity is
anchored by the separately recorded archive SHA256; it is not a signature or a
security boundary against hostile edits to the manifest and code together.
The runner checks eight ABI dependencies, not every transitive GPU component.
The qualified native runtime and its retained fallbacks remain prerequisites.

`probes/` retains the accepted B07.2/3 contracts and B07.6 public replay binaries.
`licenses/` contains MIOpen and CK licenses. `sources/` contains exact matching
headers/facade/CK library sources, accepted inputs, private/native config, build
script and source revision. This is a focused source snapshot; rebuilding still
requires the matching external ROCm compiler/SDK, third-party headers and CK static
instance archive (hash pinned in manifest). It is not a complete ROCm rebuild kit.
The build script's `/workspace` mounts are the recorded B07.3 build layout; remap
the supplied accepted inputs as `/workspace/inputs`, writable output as
`/workspace/output`, and the matching source/native SDK/cache paths accordingly.
No source tree, build or cache is deleted by this artifact.
