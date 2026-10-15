# Using bc250-torch for training and persistent files

> **Historical snapshot notice:** this file preserves a previous milestone.
> Read [flow.md](../../flow.md) and the
> [current memory inventory](bc250_memory_inventory.md) before acting on its
> active-release claims or unchecked tasks. Newer qualified scopes supersede
> older unsupported/pending rows; retain the original evidence.

The reviewed daily-use release now supplies `bc250-train train.py`, which selects
the independently frozen Transformers/PEFT/Accelerate packages. The original
developer venv remains available. See [bc250_release_20261013.md](bc250_release_20261013.md).

`bc250-torch` runs the validated Torch/ROCm image through rootful Podman on
`.104`. Python itself runs as `nik` (UID/GID 1000). The ROCm and Torch runtime
files are read-only; your project and data files use host bind mounts.

## Attention selection on .104 (updated 2026-10-15)

The incorrect `1Torch was not compiled with memory efficient attention` warning
is fixed in the current development runtime. The installed `finetune-n7`
`train_lora.py` and `chat.py` also use the supported `sdpa_kernel` API; training
backward stays inside the context for checkpoint recomputation. These corrections
apply by default. They do not require enabling the new tiled backend.

**Math attention remains the default.** The memory-saving GPU tiled backend is
opt-in. For those two installed scripts, append `--attention-backend tiled` to
your existing command, **after the script path**, keeping all model/data/output
arguments. For example, to start chat with the optional backend:

```sh
bc250-train /home/nik/finetune-n7/scripts/chat.py \
    --model /home/nik/finetune-n7/models/Qwen3-1.7B-Base \
    --attention-backend tiled
```

Use `--attention-backend math` or omit the flag to retain the default. The flag
belongs to these application scripts, not the `bc250-torch`/`bc250-train` launcher;
it does not automatically work in arbitrary third-party scripts.

For your own Transformers application, explicitly register the backend before
loading the model:

```python
from torch.nn.attention._gfx1013 import register_transformers_attention

register_transformers_attention()
model = AutoModelForCausalLM.from_pretrained(
    model_path, attn_implementation="bc250_tiled", dtype=torch.bfloat16
)
```

This prototype supports eager first-order FP32/FP16/BF16 attention, grouped heads,
head width up to128 and sequence length up to4096, with the documented mask
contracts. Attention dropout, mask gradients, higher-order gradients and compiled/
graph operation are unqualified or unsupported. Ordinary functional SDPA keeps its
math fallback. Tiled attention supports CUDA autocast. Causal head128 kernel fixtures improved
with tile skipping, but this is not a whole-model speed guarantee. Bounded real
Qwen3-1.7B LoRA references at512/1024tokens passed; full training/convergence remains
unqualified, and the whole-model1024-token peak stayed unchanged in our fixture.
See the [training follow-up](bc250_training_expansion_results/README.md) and the
[attention report](bc250_attention_results/README.md) for references, measurements,
exact contracts and rollback information.

## Resume state and crash diagnostics (updated 2026-10-15)

The installed `train_lora.py --resume CHECKPOINT` keeps adapters trainable and
restores GPU RNG from the required CPU byte tensor. New checkpoints also save
scheduler, Python RNG and max-steps shuffle/remaining-index state. Existing
old-format checkpoints remain usable, with warnings for unavailable exact replay
state. Retain matching schedule arguments; restoring a completed cosine schedule
does not create a new learning-rate plan. Bounded new/legacy fixture resumes and
separate fresh-process AMP/optimizer/RNG continuation tests passed; this is not
universal checkpoint/model compatibility qualification.

The launcher defaults `PYTHONFAULTHANDLER=1` so future fatal Python faults produce
tracebacks in your logs. Normal core-dump policy stays unchanged. To opt out
explicitly, use launcher option `--env PYTHONFAULTHANDLER=` before the script path.
The value `0` still enables Python's handler because it is a nonempty string.
The historical CPU segfault remains unresolved; see the follow-up report above.

## Paths and persistence

Host paths retain their names inside the container. Your home directory, current
working directory, and an external script's parent directory are mounted
read-write automatically. The working directory remains the directory where you
called the launcher; it does not change to the script directory.

| Scenario                                    | What happens                                                                                              |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `torch.save(..., "checkpoints/model.pt")`   | Writes under the caller's host working directory; survives container exit                                 |
| `/home/nik/models/model.pt`                 | Uses the same absolute path on the host and inside Python                                                 |
| `Path.home()` and `os.path.expanduser("~")` | Resolve to `/home/nik`                                                                                    |
| Script outside the current directory        | Its directory is mounted at its original path; sibling imports/files remain accessible                    |
| `Path(__file__).parent / "config.json"`     | Reads the host file next to the script                                                                    |
| Host edits a mounted file between runs      | The next container sees the host edit                                                                     |
| Container edits a mounted file              | The host sees the edit immediately; ordinary file locking/atomic replacement rules still apply            |
| Files named with spaces or Unicode          | Supported; quote paths in the shell                                                                       |
| Symlink into another directory              | The target must also be mounted; home and cwd are automatic, external targets need `--mount`/`--mount-ro` |
| `/tmp` and unmounted container directories  | Temporary; use an explicit host mount if their contents must survive                                      |
| New Python process/container                | Mounted checkpoints, downloads and extra packages remain; Python in-memory state starts fresh             |

Creating a directory inside a mounted directory creates it on the host too. An
explicit mount source must already exist. The launcher does not guess dataset
paths from script arguments or scan your code for paths.

For files on an external disk or outside the automatic mounts:

```bash
mkdir -p /mnt/models/checkpoints
bc250-torch --mount-ro /mnt/datasets --mount /mnt/models train.py \
  --dataset /mnt/datasets/train.json --output /mnt/models/checkpoints
```

Both explicit options support an existing file or directory and preserve its
absolute path. `--mount-ro` prevents writes through that mount. Automatic cwd/script mounts
inside a read-only parent inherit its mode. An explicit `--mount` can make a
specific output subdirectory writable. Filesystem
permissions still apply; the container gets the user's supplementary groups.
Existing filesystem access on the host does not automatically mount every disk.
A symlink to `/mnt/datasets` also needs that target mounted.

## Launcher options and Python arguments

Launcher options precede Python/script arguments. Once Python arguments begin,
remaining options belong to Python or your script. An optional `--` ends launcher
option parsing.

```bash
bc250-torch --launcher-help
bc250-torch -u ~/projects/training/train.py --epochs 10
bc250-torch --mount-ro /mnt/data -- -m my_project.train --data /mnt/data
bc250-torch -c 'from pathlib import Path; print(Path.cwd(), Path.home())'
printf 'print("stdin works")\n' | bc250-torch -
bc250-torch --shell
bc250-torch --shell -c 'ls "$HOME/models"'
```

Shell mode keeps the validated runtime PATH and skips host Bash startup files.

`-h` and `--help` are forwarded to Python; use `--launcher-help` for launcher
help. The launcher returns Python's exit status. Doas may request your password
before creating the container. Regular host authentication requirements also
apply to piped or background invocations.

## Checkpoint save and resume

Save model/optimizer state rather than keeping it only in the container's memory:

```python
from pathlib import Path
import torch

checkpoint = Path("checkpoints/latest.pt")
checkpoint.parent.mkdir(parents=True, exist_ok=True)
temporary = checkpoint.with_suffix(".tmp")
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": torch.get_rng_state(),
        "gpu_rng": torch.cuda.get_rng_state(),
    },
    temporary,
)
temporary.replace(checkpoint)
```

The temporary file is in the same directory as the final checkpoint, so readers
see the old or new file after replacement. This handles ordinary save completion;
it does not promise power-loss durability. For concurrent writers use unique
paths or an application-level lock.

Resume in another `bc250-torch` invocation:

```python
state = torch.load("checkpoints/latest.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
torch.set_rng_state(state["rng"])
torch.cuda.set_rng_state(state["gpu_rng"])
step = state["step"]
```

Our gate saves GPU model/Adam state, restores it in a new container and continues
training. It also verifies CPU loading with `mmap=True`. Scheduler, AMP scaler,
sampler and application-specific state should be saved if your training needs
them; the launcher cannot reconstruct state that your program did not save.

The host can copy, back up or transfer checkpoints with normal tools:

```bash
ls -lh ~/projects/training/checkpoints
scp ~/projects/training/checkpoints/latest.pt another-host:/path/to/backups/
```

Python checkpoints need compatible loading code; bind mounts do not change the
checkpoint format. Weight/checkpoint/model-library compatibility is separate
from the file transport validated here.

## Downloads, model caches, credentials and network

The launcher uses host networking, so code can download files and reach services
on the host's `127.0.0.1`. Files downloaded under your home/project directory
persist. Default home-based Hugging Face, Torch Hub and pip cache locations are
on the host; no additional mount is needed for those locations.

Common `HF_*`, `HUGGINGFACE_*`, `WANDB_*`, `MLFLOW_*`, HTTP proxy settings,
`TOKENIZERS_PARALLELISM`, thread limits, device-visibility variables,
`XDG_CACHE_HOME` and `TORCH_HOME` are carried across `doas`. Inherited names are
listed in `INHERITED_NAMES`/`INHERITED_PREFIXES` in `bc250_torch.py`. Other values
can be passed explicitly:

```bash
export MY_TRAINING_SETTING='value with spaces'
bc250-torch --env MY_TRAINING_SETTING train.py
bc250-torch --env HF_HOME=/mnt/models/hf-cache --mount /mnt/models train.py
bc250-torch --env-file ~/training.env train.py
```

Environment files contain literal `NAME=VALUE` lines, with blank lines and `#`
comments allowed. They do not evaluate shell expressions or remove shell quotes.
For example, write `MY_SETTING=value with spaces` directly. Values must be single
lines. Export tokens on the host or use a permission-restricted environment file;
`--env NAME` reads the caller's value without putting it in the Podman argument
list. The elevation handoff and Podman env file use private temporary files and
are cleaned up after the command; their contents are not logged.

Values selecting the validated runtime, including `HOME`, `PATH`, ROCm library
paths, SDMA and compiler cache locations, are controlled by the launcher. Their
replacement is rejected. Other library-specific model-cache locations can be
set normally, but external paths still need a corresponding mount.

The download gate uses a real HTTP transfer from a temporary host loopback server,
then reads and hashes the downloaded file in a subsequent container. It does not
establish connectivity or credentials for every external provider.

## DataLoader workers and shared memory

The default `/dev/shm` limit is **1 GiB**. This is a limit rather than an up-front
allocation. It avoids Podman's usual 64 MiB default for training workers; the
option is documented in the [Podman run reference](https://docs.podman.io/en/latest/markdown/podman-run.1.html#shm-size-number-unit).

```bash
bc250-torch --shm-size 2g train.py
```

The board shares physical memory between CPU and GPU. Larger limits do not add
physical RAM; choose batch size, worker count and prefetching to fit the workload.
The gate touches 80 MiB of shared memory and checks a two-worker DataLoader with
`multiprocessing_context="spawn"`, persistent workers, pinned memory and exact
CPU-reference batches over two iterations. Keep the normal Python
`if __name__ == "__main__":` entry-point guard when using spawn.

## Extra Python packages

The validated image contains an immutable Python environment. Install additional
packages into a mounted directory if they must survive container exit:

```bash
mkdir -p ~/bc250-python-packages
bc250-torch -m pip install --target ~/bc250-python-packages --no-deps PACKAGE
bc250-torch --env PYTHONPATH="$HOME/bc250-python-packages" train.py
```

This workflow was tested with an offline local pure-Python package, imported in a
new container. `--no-deps` keeps pip from replacing the validated Torch/Triton and
numerical dependencies; install required compatible dependencies explicitly.
Do not use Alpine-built binary Python extensions in the glibc container. An
extra package can shadow an installed package through `PYTHONPATH`, so its own
compatibility still needs checking. Packages installed only into a disposable
container layer do not survive exit.

## Interruption and detached jobs

Terminal Ctrl-C is passed to the foreground runtime. The host integration gate
also checks SIGINT and root-issued SIGTERM with an application handler that writes
a final host-visible file before exiting. A graceful checkpoint on interruption
requires a handler in your training program; abrupt termination cannot complete a
save automatically.

Rootful Podman is started through `doas`. For administrative termination of a
detached rootful runtime use root permissions; killing just an unprivileged
wrapper is not a reliable way to stop the container. Authenticate normally when
starting detached work and use persistent host paths for its logs and outputs.
Distributed training and arbitrary third-party launchers are separate runtime
qualifications; this file workflow does not establish RCCL/multi-GPU support.

## Verification and recovery

Thirteen host integration scenarios passed on 2026-10-13. The actual `nik`
elevation handoff also passed environment and GPU-checkpoint checks, piped
stdin and interactive shell mode. The user matrix script passed after installation.
Read-only parent/cwd inheritance and an explicit writable output exception were
also verified on the host.

The host integration runner is `build_tools/bc250/validate_launcher_on_host.py`.
It can test a staged launcher with `--launcher PATH` before installation. Evidence
is stored under the `.104` workspace's `logs/launcher-io/`; sample checkpoints
are under `launcher-io-validation-*/training project/checkpoints with spaces/`.

The previous installed launcher is preserved as
`recovery/bc250-torch-before-training-io-20261013` in the workspace.
Launcher source revision/hash and evidence are recorded separately in
`/opt/bc250-stack/launcher.json`. Existing release-home compilation caches were
merged into the host home cache without overwriting existing files.
The runtime's `/opt/bc250-stack/current` and `/opt/bc250-stack/previous` symlinks
are independent of the launcher update. See [bc250_port.md](bc250_port.md) for
runtime recovery and [bc250_full_support_roadmap.md](bc250_full_support_roadmap.md)
for remaining ROCm work.

Real pretrained full fine-tuning, FP16/BF16 LoRA and fresh-container training
resume are now qualified for the small recorded workload. Use the isolated
consumer environment and examples in [bc250_training_memory.md](bc250_training_memory.md).
