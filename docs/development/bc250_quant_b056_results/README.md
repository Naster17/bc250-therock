# B05.6: optional W8 runtime package

The .104 daily and explicit `rccl-single` runtimes now contain `bc250_quant 0.1.0`.
Public imports work without a helper-directory search path:

```python
from bc250_quant import FrozenW8Linear, FrozenInt8Weight, pack, linear
```

Use these imports in scripts launched by `bc250-torch` or `bc250-train`.
Conversion remains explicit and requires an individually frozen standard GPU
`nn.Linear`. Existing Torch models/dispatch/defaults are not converted.
[Usage](../bc250_torch_usage.md#explicit-frozen-w8-module-and-checkpoints-b055).
Tooling revision: final `abd8a4e04ecaa2584129a621ee4322295ebbd542` (full revision in verification.json); initial package tooling `aee53dcc0` retained.

## Package and installation

A reproducible pure-Python wheel is built from pinned qualified source hashes.
`bc250_quant/_w8.py` exactly copies B05.4 helper6832ff9e; `modules.py` copies the
B05.5 modulea18e988f with only its helper import replaced by a relative import.
The package exports the two classes and `pack`/`linear`, with version metadata,
source provenance and the MIT license. Wheel RECORD hashes, installed paths,
exports/version, absence of import-time dispatch hooks and source equivalence
are checked. Requires-Dist is `torch`; installation uses `--no-deps --no-index`
with no network. It neither installs nor admits TorchAO/bitsandbytes/NF4.

The image derives from the existing image506f3b26, retaining its full layer prefix.
Only the wheel and its pip installation are added. Host-mounted native Torch/ROCm
and old helper files stay unchanged. Image building uses RAM1GiB, CPU quota2,
no network/pull,180s deadline; no native build or ccache operation. Two initial
staging attempts refuse unsupported `podman build --cpus` before any build; the
first retry mistakenly retained that flag. Final r3 uses supported CPU period/
quota flags. Original failure logs/contexts remain. No GPU/numerical assertion
was relaxed and no failed image was selected.

Candidate manifests independently mirror daily/RCCL prefixes and change only
image selection plus `quant_b056` provenance metadata. Both candidate and installed
acceptance preserve native library identities, profile settings, boot fallbacks,
compatibility links and four AOT DBs. Daily NCCL stays off/Gloo on; explicit profile
remains bounded one-GPU/rank1 RCCL. Math SDPA/SDMA0/heap-off/N3.2 unchanged.

## Tests and scope

817 candidate +817 ordinary installed checks,32 guarded gates in total:

| Checks per phase | Coverage |
| ---: | --- |
|8 | Both cohorts: installed package path/version/exports, RECORD hashes, exact source identity, native device and retained FFT/sparse/vision/audio/codec imports. |
|142 | Both cohorts: module conversion/dtype/device/layout/output/input/bias first derivatives, strict state validation and two-step owned adapter AdamW/resume against independent CPU references. |
|142 | Both cohorts: unchanged original W8 recipe, including independent CPU packing/output/first gradients, finite differences, private stream, byte/tie/zero-row cases, AMP/refusals and native GPU trace. |
|11 | Daily/opt-in consumers and backend identity. |
|7 | Retained core GPU correctness. |
|245 | Retained explicit BF16 FFT promotion references/first gradients. |
|253 | Retained bounded CSR dense-LU solve references/first gradients/refusals. |
|9 | Actual pre-package B05.5 checkpoint files, three dtypes × three bias policies: public output/input/bias gradients exactly match legacy, and saved public state reloads through legacy. Installed compatibility uses `bc250-train`. |

The public tests reuse unchanged B05.1/B05.5 reference suites by binding their
helper imports to the installed public implementation. Installed file identities
are checked before that binding, preventing a helper implementation fallback.
Actual checkpoint compatibility additionally compares public and retained legacy
implementations directly. Owned fixtures only; no user applications or data.

Pure-Python wheel construction is byte-reproducible, with local RECORD/license/
core checks;29 existing BC250 unit tests and selected Ruff checks pass. Kernel
logs match per gate and through candidate/installed/export; all accepted guards ACK.
An initial installed consumer gate incorrectly used `bc250-torch`, where the
training-only `accelerate` package is absent; it refuses with ModuleNotFoundError
after the five W8 gates (146 checks) pass. Candidate consumers use the internal
training context and passed. Both manifests are restored under the exclusive
lock; the public consumer runner is corrected to `bc250-train`. The r2 run passes both cohorts' W8/consumer checks (303), then the retained core
fixture also imports training-only `transformers` and refuses under the plain
Torch launcher. The controller automatically rolls back both manifests again.
All retained consumer/core/FFT/sparse fixtures now use `bc250-train`, matching
their existing internal candidate environment; W8 public API suites still use
`bc250-torch`. Both suites repeat as r3 with identical image/wheel and unchanged
reference assertions. Initial candidate808+808 and installed146+303 completed
checks, both failed gates, original tools and both rollback records are preserved.
All81 guards (32 accepted plus49 previous-trial guards) are acknowledged; no
numerical contract or tolerance changes. No own tests/containers/armed guards
remain at handoff.

This is packaging/admission of the prior bounded eager W8 capability, not new
native arithmetic, broader quantization, model quality, peak-memory, training
speed, higher gradients, compiled W8/graphs or stability qualification. Native
weight-only backward/packed INT4 refusals remain. The historical Python CPU
segfault remains unresolved, distinct from fixed VM-heap/CSR/hostcall defects.
`.105`, user applications, existing sources/builds/caches/checkpoints/images and
kernel/boot controls are untouched.

## Reproduction and recovery

Run the stage/qualification tools from `WS/helpers` as root on .104. Candidate
and ordinary tests hold the daily shared lock; internal candidate testing selects
an isolated release, without publishing a profile. Promotion/acceptance/rollback
use the daily exclusive updater. Never nest an ordinary launcher inside it.
Staging labels are single-use evidence snapshots, not general rerun commands.

Both old runtime manifests are preserved in `WS/recovery/quant-b056-before`.
The locked two-manifest update validates both generations first, journals each
change and restores its own partial changes on Python exceptions. It is not an
atomic multi-file transaction across sudden host failure. If ordinary acceptance
fails, release its shared lock and restore through the exclusive updater.
Identity-guarded rollback (exercised after the initial consumer-runner refusal):

```sh
WS=/home/nik/bc250work/rocm-10.2-gfx1013
doas python3 "$WS/helpers/update_dev_prefix.py" -- \
  python3 "$WS/helpers/deploy_quant_b056_on_host.py" --rollback
```

This restores both prior image506f3b26 manifests exactly. New/old images and
candidate releases remain; legacy helper imports still work. Runtime/image/wheel/
source/boot and preservation identities: [verification.json](verification.json).
Raw reports: `WS/logs/quant-b056-{candidate,installed}-r3/final.json`, plus the
compatibility gates, stage-r3 and deployment record. Local archive:
`kernel/artifacts/quant-b056`; [archive.json](archive.json). It contains raw
fixtures/logs/guards, wheel, image metadata, source tools and recovery manifests.
Archive4,165,410B SHA256`d57be371cd444afd56739af693c35c5477a1843125dde6475af944275a09aef5`;
790 raw log/fixture files18,442,720B, nine verified tools, wheel/image metadata/
recovery and81 guards.457 protected hashes/four AOT DBs/links retain expected
identities with the two manifest updates/recovery explicitly accounted for.
Daily runtimeSHA5a3a4c2f/profile6101835b; native Torcha524 source remains.
Prior reference source is additionally retained in B05.1/B05.5 artifacts and
protected hashes. Memory snapshot is synchronized to .104 only.

**Next B06.1:** audit ONNX export/runtime package and execution-provider feasibility
against the pinned Torch/gfx1013 stack; select a bounded isolated export/CPU-reference/
GPU-interoperability qualification. Preserve the new W8 package and native defaults;
do not claim a GPU ONNX execution provider from CPU export/runtime availability.
