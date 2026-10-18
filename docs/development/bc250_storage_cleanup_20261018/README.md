# BC-250 generated-artifact cleanup — 2026-10-18

User explicitly requested another cleanup of old build/junk artifacts on .104.
Recovered **12,839,837,696 bytes (11.96GiB)** by available-space change, from
**1.08GiB to13.04GiB free**, filesystem100%→97% used. This retires only inspected
obsolete generated outputs, superseding older retention statements for these
specific artifacts. It changes no ROCm/Torch implementation or support result.

Implementation: TheRock `780e67ceb`, `build_tools/bc250/cleanup_storage_refresh.py`,
`validate_storage_refresh_runtime.py`, `finalize_storage_refresh_on_host.py`.
Executed under the existing exclusive development-prefix lock. Exact plan and
8,921-action journal are `plan.json.gz`/`applied.json.gz` here (uncompressed originals
on .104); `final.json` seals source hashes, preservation and GPU results.

| Removed | Count | Allocated file bytes before unlink |
| --- | ---: | ---: |
| Superseded Torch .o/.a/.so outputs under WS/pytorch-{work,a04,sparse-followup}/build | 8,801 files | 5,326,098,432 |
| Old Triton ELF/test/tool binaries and compiled test objects under WS/triton-work/build/{bin,test} | 21 files | 5,379,858,432 |
| Our synthetic CLI checkpoint adapter/state weights and duplicate tokenizers | 90 files | 2,124,718,080 |
| Explicit stopped project build containers (A04/sparse-followup/A09) | 9 containers | Small writable layers; image/shared layers unchanged |

Synthetic files are only under WS/logs/{a06-a07-installed-cli,
a06-a07-installed-cli-r3,a06-a07-installed-cli-r4,profiler-a09-candidate-cli,
profiler-a09-installed-cli}/synthetic-cli. These are our generated test checkpoints,
not user LoRA runs. Numerical/configuration/results logs and checkpoint metadata
remain; old deleted fixture payloads must be regenerated under a new test label.

No directory-tree removal: source, generated source, build metadata/configuration,
directory paths and symlinks remain. Missing generated build targets can be rebuilt
from their retained sources/configuration and compiler cache. Each affected old
build directory has GENERATED_ARTIFACTS_RETIRED_20261018.json pointing to the journal.
The current A09 Torch incremental build is intact. The immutable third_party source
at pytorch-sparse-followup/third_party, required by future A09 builds, is intact.
Triton build libraries/object cache/configuration and python/triton native output
remain; only the selected bin/test files were unlinked. This is not a full build
expunge or a compiler-cache wipe.

## Scope checks and preservation

Before removal: current/previous link identities checked, no running Podman job,
no live process cwd/exe/fd/maps reference into candidate scopes, no inbound symlinks
from project runtime/source/helper/environment roots. Every file is an explicit
non-symlink regular path with checked device/inode/size/mtime; exact stamps are in
the plan. No sources, caches, real user models/data/checkpoints, recovery copies,
source archives, existing Git histories or user-owned src were selected.

101 active/old/frozen runtime/configuration/product hashes match before/after,
including existing82 N2.11 protected items, additional frozen/current runtime files,
old Torch package extensions/source stamps and retained Triton native output.
All images and image tags remain unchanged. The user neural-p1 container's
ID/image/status is unchanged. Nine stopped project build containers were removed
by exact inspected IDs, without force/volumes/global prune; redacted log tails are
retained on host and in ignored local evidence. No package uninstall, user-cache
sweep or local/other-project/.105 cleanup.

Active ROCm57.3GiB build, all source trees, active Torch A09 source/build/binaries,
working ccache/Triton/Inductor caches, software fallback and kernel r2 boot fallback,
/opt compatibility installs/links, release payloads/metadata, training environments,
real datasets/checkpoints/models and experimental recovery snapshots remain.
No current/previous/torch/rocm link or runtime metadata change. Kernel/boot/BIOS/
power/SDMA0/heap-off/math defaults are unchanged. Historical Python CPU fault
remains unresolved and distinct from the fixed VM-heap/CSR issue.

## Bounded runtime checks

Five guarded GPU processes all pass8 checks each (40 group executions): current
before, current after, frozen fallback after, plus current/fallback fresh-JIT
checks. Seven retained CPU-reference groups cover FP32/BF16 GEMM outputs/gradients,
math SDPA D32/D256 outputs/QKV gradients, layernorm/FFT gradients and exact RNG
replay, plus consumer imports. The eighth check runs a nondefault-stream Triton
vector kernel with exact CPU comparison. Final two runs use a new kernel signature
and distinct label-derived constexpr biases to force new compilation/cache keys,
checking compiler availability after old test/tool binary removal. Both resolve
Triton's native library inside the retained runtime image, not removed build files.

All five600s guardians acknowledged, exit0. dmesg is byte-identical across the
entire cleanup, not only within individual test windows. Same r3 kernel7.2.10,
boot e3ed129b-9c41-4d0c-ab84-5ab54720bb8f; no active agent job/container/guard deadline.
These short checks do not establish universal correctness or long-run stability.
No native build or cache reset; Python syntax/Black/diff checks pass, no full CI claim.

Raw .104 evidence: WS/logs/storage-refresh-20261018 and storage-refresh-* gate
labels; ignored local kernel/artifacts/storage-refresh-20261018 retains the full
archive, including kernel logs and redacted container tails. Planner/applicator
is a restricted dated maintenance helper, not a general garbage collector.

Next action returns to unchanged **N2.12**: profile warm-session right-vector
recovery and prototype bounded native triangular back-substitution, retaining
Python/N2.6 reference, fixed separation/rescaling/unitnorm/residual/CPU gates,
cap16 and production host-LAPACK default.
