# Explicit FP32/FP16 MIOpen DL activation on .104 (B07.13)

B07.13 installs and activates the qualified **B07.12 coherent FP32/FP16 CK +
repaired-MIOpen** artifact as the optional `bc250-ck` command, replacing the
B07.7 FP32-only registration. The native ROCm/Torch payloads, production
architecture whitelist and ordinary `bc250-torch`/`bc250-train` defaults are
unchanged. The B07.7 FP32 package/tool/version records are **retained** as the
live-rollback target. .105 is not updated.

```sh
bc250-ck --launcher-help
bc250-ck --ck-info
bc250-ck --check
bc250-ck --runtime-profile rccl-single --check
```

`--ck-info` describes registration; `--check` checks the actual mapped container
libraries (27 identities, including the private repaired MIOpen), the native GPU
architecture and the plugin API. It does not benchmark a model.

Run a Python script through the ordinary qualified runtime with the plugin enabled
for that process:

```sh
bc250-ck my_model.py --your-option value
bc250-ck --runtime-profile rccl-single my_model.py
```

Launcher options must precede the script. Mounts, `--env`, `--env-file` and
`--shm-size` work as in `bc250-torch`. Bare environment variables and arguments
containing spaces survive ordinary-user `doas` elevation. The command accepts
Python files; `--shell`, Python `-c` and `-m` are not supported. It uses the
qualified runtime interpreter and does not automatically add the frozen
`bc250-train` consumer overlay.

## Format2 worker activation

Unlike the B07.7 same-process entry, the B07.12 entry is **format2**: the parent
verifies the payload bytes, then spawns a **fresh worker** whose `LD_LIBRARY_PATH`
puts the package's private repaired MIOpen (`runtime/libMIOpen.so.1`) ahead of the
runtime SDK. The worker verifies the 27 mapped library identities, native
`gfx1013:xnack-` and the API5/66 exports, then runs the script as `__main__` with
its argv/import directory. The parent stays alive; there is no `exec` replacement
after HIP initialization. The private-MIOpen loader ordering is impossible to
retrofit into an already-running loader, which is why the fresh worker is required.
Do not add a global `MIOPEN_CK_LIB_PATH` or `LD_LIBRARY_PATH` to application
environments; the entry sets them inside the worker.

## Tuning records and scope

The package examples contain genuine records for **2 FP32 + 6 FP16** keys per
cohort. To copy them into a **new** owned directory without merging existing data:

```sh
bc250-ck --seed-db "$PWD/miopen-dl-db"
mkdir -p "$PWD/miopen-dl-kernels"
bc250-ck --env MIOPEN_USER_DB_PATH="$PWD/miopen-dl-db" \
  --env MIOPEN_CUSTOM_CACHE_DIR="$PWD/miopen-dl-kernels" my_model.py
```

For the RCCL example records, put `--runtime-profile rccl-single` before
`--seed-db`. The seed command refuses an existing destination. Native tuning/search
remains MIOpen's own machinery; the wrapper does not force a solver or fabricate a
general shape heuristic.

The plugin supplies native **FP32/FP16 packed2D forward**, exact alpha1/beta0,
restricted descriptors/pointers/workspace. BF16, mixed types, casts,
transpose/fused groups and unsupported shapes refuse. The bundled HIP-direct
gfx1013 forward retains FP32 accumulation. Backward kernels are not implemented by
the plugin; retained native solvers perform backward. Neither universal correctness
nor graph/concurrency coverage follows from these fixtures. B07.11's one-of-six
1.17–1.18× fixture gain remains a narrow historical measurement; this package forces
no global solver choice.

## Installation identities and recovery

- Artifact: `/opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b0712`.
- Registration: `/opt/bc250-stack/sdks/miopen-dl.json`; version record
  `bc250-miopen-dl-gfx1013-b0712.json` retained independently.
- Wrapper/dependencies: `/opt/bc250-stack/tools/ck-b0713`; public command
  `/usr/local/bin/bc250-ck`.
- Accepted archive SHA256:
  `a0d47ed4ba4df354cc7ea898a285eefa051f49693667386567ea392f66ddd91e`.
- Manifest SHA256:
  `52db5d4262bd29e6307b13a5862cc0652f0e0b6f9c8509761d03d39965e02846`.

Both runtime metadata hashes/images, the exact manifest/payloads and the 27 mapped
library identities are pinned. A native runtime/image upgrade can intentionally make
activation refuse until requalified; do not bypass that refusal by editing hashes.
The registration is root-controlled; launch holds the shared runtime lock for
validation and the application's lifetime. Install/activate/unregister/rollback use
the exclusive update lock. Never invoke an ordinary launcher while holding that lock.

Because the b0713 wrapper refuses the old b077 registration, **activation and
rollback move both `miopen-dl.json` and the `/usr/local/bin/bc250-ck` symlink
together** under the exclusive lock. The deploy captures the prior registration +
wrapper target to `miopen-dl-b0713-rollback.json`.

```sh
stack_ws=/home/nik/bc250work/rocm-10.2-gfx1013
# install (stages package/tools/record; does not switch activation)
doas python3 "$stack_ws/helpers/deploy_ck_plugin_b0713_on_host.py" install \
  --archive /home/nik/bc250work/ck-plugin-b0713-inputs/bc250-miopen-dl-gfx1013-b0712.tar.gz \
  --receipt "$stack_ws/logs/ck-plugin-b0713-install-r1/final.json"
# activate (switch registry + wrapper to b0712)
doas python3 "$stack_ws/helpers/deploy_ck_plugin_b0713_on_host.py" activate \
  --receipt "$stack_ws/logs/ck-plugin-b0713-activate-r1/final.json"
# disable (activation refuses; data retained)
doas python3 "$stack_ws/helpers/deploy_ck_plugin_b0713_on_host.py" unregister \
  --receipt "$stack_ws/logs/ck-manual-unregister-$(date +%Y%m%d-%H%M%S).json"
# rollback to the retained B07.7 FP32 activation
doas python3 "$stack_ws/helpers/deploy_ck_plugin_b0713_on_host.py" rollback \
  --receipt "$stack_ws/logs/ck-plugin-b0713-rollback-r1/final.json"
```

Re-activate with `activate` and a new receipt path. Ordinary Torch/train commands
keep their defaults throughout. Use the versioned helper and retained identity
records rather than deleting files.

[Installed qualification](bc250_ck_plugin_b0713_results/README.md),
[candidate artifact scope](bc250_ck_plugin_b0712_results/README.md),
[developer architecture/environment guide](../../../BC250_DEVELOPER_GUIDE.md).