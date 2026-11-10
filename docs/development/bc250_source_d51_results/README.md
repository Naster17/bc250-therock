# D51: selected native build-input disposition and portable evidence

Completed 2026-11-10 for this inventory/replay scope. PLAN-07 remains open for
payload assembly and release closure. .104 runtime and .105 remain unchanged.

## Missing dependencies: a configuration-specific disposition

All eight missing D50 branches sit under aiter, flash-attention or mslk. Both
retained native builds explicitly disable FLASH_ATTENTION, MEM_EFF_ATTENTION,
ROCM_CK_SDPA, ROCM_CK_GEMM and MSLK. The complete compile databases contain3,239
entries for daily and3,241 for RCCL; neither commands/includes nor the retained
Ninja graphs mention these dependency families. Pinned native Torch CMake
sources provide the matching feature guards, including the CK/aiter include guard.

These eight branches are **excluded for this exact recorded configuration**;
zero missing-branch fetches are required here. They remain unexpanded and must
be admitted for another configuration. This does not claim a new full clean build,
complete recursive source checkout, or optional CK/flash/MSLK hardware support.
No initialized checkout, cache or unsupported backend was changed.

## Original build provenance recovered

The retained B05.2/B05.3 build records and scripts identify the original immutable
build image, still present on .104:
`28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06`.
It is distinct from the current runtime image:
`1bb4ac0ed3766d9c70431c1d5119e03cd4d10e710f4c04cd54254b5b87cd45d5`.
Only selected original image/command fields are exported; old application-bound
preservation baselines are not copied into this package.

Both retained scripts explicitly export PYTORCH_BUILD_VERSION=2.13.0a0+gfx1013,
PYTORCH_BUILD_NUMBER=1 and PYTORCH_ROCM_ARCH=gfx1013, and use the authorized pip-only
editable build. This closes the version-input inventory gap from D50; D50 correctly
said these values could not be inferred from version.py alone. Generated
`git_version='Unknown'` remains unchanged. Native source a524980 and later Python
source8d8b18d remain distinct; no metadata rewrite invents a native rebuild.

The original image and runtime were probed with no network, GPU nodes or Torch
import. HIP Clang comes from the retained read-only mounted ROCm build tree,
not from either image; its hash is measured now, not a retained digest proving
byte identity at the time of the historical Torch builds. Seven actual tool executable paths/resolutions/hashes/version outputs are
bound in each image, with35 versus49 installed Python distributions and their
available RECORD metadata hashes. Package metadata is an inventory, not a checksum
verification of every installed package file or a complete license review.

| Input | Observed selected identity |
| --- | --- |
| GCC / G++ | Ubuntu13.3.0 |
| HIP Clang | AMD24.0.0git, LLVM7667f52dd687899256a3285ab0bce3c53bc73394 |
| Actual CMake | 3.31.12 |
| Ninja | 1.13.0.git.kitware.jobserver-pipe-1 |
| ccache | 4.9.1 |
| CPython | 3.12.3 |
| Native build Python packages | setuptools81.0.0, numpy2.5.1, packaging26.2, PyYAML6.0.3, requests2.34.2, six1.17.0, typing-extensions4.16.0, pip26.1.2, wheel0.47.0 |

A concrete provenance trap is now detected: installed Python `cmake` distribution
metadata says4.4.0, but `/opt/bc250-tools/cmake` and the cache-selected executable
are byte-identical CMake3.31.12. The final probe checks the selected path's SHA
against that actual tool. Future packaging must preserve the executable/resolver,
not select a tool merely from pip metadata. An initial supplemental inspection
assumed the resolver was text and refused UTF-8 decoding of its ELF bytes; a
read-only binary hash check resolved that assumption. R1 is preserved; final R2
adds selected-path hashing and passes both image probes again.

The retained313,852,959-byte Triton wheel is SHA256
`2d65d6fefe359116521640e3cc2257bd03b4b24f6877fc8e28946ee9c57544d8`.
The nine pyproject build-system requirements, toolchain file, original recipes,
raw CMake caches/Ninja/compile commands/version files, five pinned native source
files and Torch license are in the portable evidence package.
[Inventory](input-inventory.json), [verification](verified-inputs.json).

## Portable package and checks

`kernel/artifacts/source-d51/portable-build-inputs-d51-r1.tar.gz`:1,130,174B,
SHA256 `ddb71d39c18d82e7b3a4acace088037ae58176c65a303b9ab8c50686c04b580b`.
[Receipt](package.json). .104 stores the same archive under tools/source-d51/,
with extracted build-inputs-r1/. No source checkout, image, network or GPU is
needed to replay this evidence:

```sh
python3 /new/build-inputs-r1/tools/verify_build_inputs_d51.py \
  --package /new/build-inputs-r1 --output /new/verified.json
```

Workstation and .104 extracted-package replays pass.144 local tests include four
new feature-drift/include-reference/version-parser cases. Black, critical
Ruff+F401 and diff checks pass. Three altered source/backend/CMake packages are
refused; updating a changed cache's recorded hash still cannot bypass the feature
guard. [Negative controls](negative-controls.json).

The image probes run read-only/network-off/pull-never as UID1000, with no device
mappings,2CPU/affinity0–3/1GiB RAM and zero container swap allowance/128PIDs/64MiB
tmpfs,120s child deadlines,100ms sampling/3GiB host+5GiB disk reserves, EX cooperative
board reservation + SH runtime lock. Only owned labeled containers can be stopped.
Final R2 cost2.474s; minimum sampled host15,084,756,992B/disk21,467,734,016B.
No cgroup peak/paging-I/O/performance claim. Supplemental binary reads used1CPU/
64MiB/32PIDs/30s and no GPU. No native build, code generation, pip install, image
pull, cache reset, source mutation, prefix update or reboot was performed.

Host tools/source-d51; evidence/source-d51-r1 and source-d51-r2 are preserved.
Controllers32552(R1) and1044(R2) finished, with source-d51-controller.log and
source-d51-r2-controller.log. Final raw export is
`kernel/artifacts/source-d51/104-evidence-r1.tar.gz`:1,084,737B,
SHA256 `542fe580f63524c7efba82afc5832b12a4d067e323dec76351d65d496b0bc788`;
host evidence/source-d51-evidence-r1.tar.gz. The export contains final R2 raw data;
R1 remains separately on host. The later portable-kit replay JSON is also retained
in host evidence/source-d51-r2/portable-kit-replay.json.

## Remaining boundary and next action

This package contains evidence/recipes, not the image, compiler binaries, full
wheelhouse or a ready-to-run Torch build. D47 parent sources and D50 available
dependency contents remain separate. Version/toolchain/package **inventory** is
now explicit; portable payload export, generated/header/link-input assembly,
license closure and a clean reproduction remain incomplete. Do not declare
PLAN-07 or whole-stack offline reproducibility complete from this metadata pass.

Next D52: admit storage for preserving/exporting original build image28ad36e6,
assemble a fresh selected-source/build-recipe kit from D47/D50/D51, and inventory
required generated/header/link inputs before any clean native rebuild. Keep .105
and active prefixes unchanged. Historical Python CPU/KFD/clocksource faults remain
unknown and separate from D49's fixed SDK sentinel crash and fixed VM-heap/D05/D21.
