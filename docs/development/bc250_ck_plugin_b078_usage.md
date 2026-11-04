# Explicit MIOpen DL activation on .104

B07.8 installs the qualified B07.7 **r2** artifact as an optional command. The
native ROCm/Torch payloads, production architecture whitelist and ordinary
`bc250-torch`/`bc250-train` defaults are unchanged. .105 was not updated.

```sh
bc250-ck --launcher-help
bc250-ck --ck-info
bc250-ck --check
bc250-ck --runtime-profile rccl-single --check
```

`--ck-info` describes registration; `--check` checks the actual container libraries,
native GPU architecture and plugin API. It does not benchmark a model.

Run a Python script through the ordinary qualified runtime with the plugin enabled
for that process:

```sh
bc250-ck my_model.py --your-option value
bc250-ck --runtime-profile rccl-single my_model.py
```

Launcher options must precede the script. Mounts, `--env`, `--env-file` and
`--shm-size` work as in `bc250-torch`. Bare environment variables and arguments
containing spaces survive ordinary-user `doas` elevation. The command accepts
Python files; `--shell`, Python `-c` and `-m` are not supported. Custom Python
packages may be supplied using the developer guide's tested overlays. The command
uses the qualified runtime interpreter; it does not select an arbitrary host venv
or automatically add the frozen `bc250-train` consumer overlay.

The runner checks package/native ABI identity, initializes HIP for the native
architecture check, then runs the script as `__main__` with its argv/import directory
in the same Python process. It avoids replacing a live HIP/KFD process with `exec`.
Do not add the old candidate directory or a global `MIOPEN_CK_LIB_PATH` to application
environments. The wrapper selects the installed path and mounts it read-only.

## Tuning records and scope

The package's examples contain genuine records for only two B07.6 fixtures. To
copy them into a **new** owned directory without merging existing application data:

```sh
bc250-ck --seed-db "$PWD/miopen-dl-db"
mkdir -p "$PWD/miopen-dl-kernels"
bc250-ck --env MIOPEN_USER_DB_PATH="$PWD/miopen-dl-db" \
  --env MIOPEN_CUSTOM_CACHE_DIR="$PWD/miopen-dl-kernels" my_model.py
```

For the RCCL example records, put `--runtime-profile rccl-single` before `--seed-db`.
The seed command refuses an existing destination. Native tuning/search remains
MIOpen's own machinery; the wrapper does not force solver137 or fabricate a general
shape heuristic. Fresh processes load valid records; invalid IDs have bounded-tested
default fallback. Keep experiments in separate user DB/kernel-cache directories.

The optional plugin supplies native scalar **FP32 packed2D forward**, exact
alpha1/beta0 only. Other groups/types are explicitly unsupported by this plugin;
existing MIOpen alternatives remain available. Torch convolution output and all
first gradients are tested with the plugin enabled, but its backward kernels are
not implemented: retained native solvers perform backward. Neither universal
correctness nor graph/concurrency coverage follows from these fixtures.

Two moderate NCHW fixtures showed1.085–1.135× public-route speed advantages in
B07.6; four other B07.5 fixtures favored HIP-direct. No whole-model/LoRA training
speedup or automatic fastest selection is promised. The native runtime's usual
JIT/cache facilities remain prerequisites; this is not a standalone ROCm SDK.

## Installation identities and recovery

- Artifact: `/opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b077`.
- Optional registration: `/opt/bc250-stack/sdks/miopen-dl.json`; the version record
  `bc250-miopen-dl-gfx1013-b077.json` is retained independently.
- Wrapper/dependencies: `/opt/bc250-stack/tools/ck-b078`; public command
  `/usr/local/bin/bc250-ck`.
- Accepted archive SHA256:
  `90f0d323d0bb4ffd3a8ae04f1acd4e17805a10d7dcb62d577840cf2120e06984`.

Both runtime metadata hashes/images, the exact manifest/payloads and eight mapped
ABI libraries are pinned. A native runtime/image upgrade can intentionally make
activation refuse until requalified. Do not bypass that refusal by editing hashes.
The registration is root-controlled, and launch holds the shared runtime lock for
validation and the application's lifetime. Install/register/unregister use the
exclusive update lock. Never invoke an ordinary launcher while holding that lock.

To disable this option, unregister it; keep package, source and cache data:

```sh
stack_ws=/home/nik/bc250work/rocm-10.2-gfx1013
doas python3 "$stack_ws/helpers/deploy_ck_plugin_b078_on_host.py" unregister \
  --receipt "$stack_ws/logs/ck-manual-unregister-$(date +%Y%m%d-%H%M%S).json"
```

Re-enable with `register` and a new receipt path. The wrapper stays present and
refuses unregistered activation; ordinary Torch/train commands keep their defaults.
Use this versioned helper and retained identity records rather than deleting files.

The rejected first B07.7 package emitted a KFD resume warning during its
exec-after-HIP lifecycle. The accepted same-process runner has clean final B07.7
and B07.8 gates; the underlying kernel return remains untraced. This is distinct
from the historical Python CPU fault and the already fixed VM-heap corruption.

[Installed qualification](bc250_ck_plugin_b078_results/README.md),
[artifact scope and failures](bc250_ck_plugin_b077_results/README.md),
[developer architecture/environment guide](../../../BC250_DEVELOPER_GUIDE.md).

The B07.12 FP32/FP16 archive now includes a matching private repaired MIOpen and
has qualified relocation/identity/reference/gradient tests. It is not yet selected
by the installed command described here. See [candidate artifact/use](bc250_ck_plugin_b0712_results/README.md);
B07.13 owns the explicit versioned upgrade and rollback qualification.
