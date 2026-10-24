# B05.5: explicit frozen-W8 module and versioned checkpoints

`FrozenW8Linear` wraps the existing strict native-gfx1013 helper in an explicit
`nn.Module`. Convert an individually selected frozen standard `nn.Linear` with
`from_float`; no application is automatically converted. Source/tool revision:
`e1543d1cf75dc679edcba1b2963e6515eb28a1cc` (full identities in [verification.json](verification.json)).
[Usage](../bc250_torch_usage.md#explicit-frozen-w8-module-and-checkpoints-b055).

## Behavior

Persistent base storage is INT8 `qweight[N,K]` plus FP32 `scales[N]`; an optional
bias is the only parameter, preserving its original trainability. A nonpersistent
zero-element buffer tracks compute dtype/device. Conversion copies bias and does
not retain or modify the floating source; the caller must release its own source
references to free that storage. Conversion of subclasses is refused so custom
forward semantics cannot silently disappear. Construct optimizers after conversion
and final dtype/device selection.

The module delegates arithmetic, finite checks, bounds and input backward to
unchanged `bc250_quant_b051.py` SHA256
`6832ff9e58104236d877a455fdfe90f11b3803198732497d6dd4ac77242fd319`.
GPU forward/input/bias first derivatives remain available; there is no integer,
scale, floating-base or higher-order autograd. FP16/BF16/FP32 inputs must match the
explicit module compute dtype, including under autocast. Bias is added after the
native output cast. Dtype/device moves preserve exact INT8 values and FP32 scales;
scales never pass through a lower precision floating dtype. CPU storage and state
loading work; forward/conversion require native gfx1013. FP64/meta moves refuse.
Arbitrary custom `_apply` transforms are not qualified.

State format v1 uses tensors plus plain dictionary metadata: format/version,
dimensions, compute dtype, bias presence/trainability. Factory loading validates
metadata and storage bounds before allocation. Loading validates exact own-module
fields, tensor layout/type/shape/device consistency, finite-positive FP32 scales
and destination metadata before copying that module's tensors. Missing fields
still refuse with `strict=False`; `assign=True` refuses. This is not a transaction
across a larger parent module. Bias finiteness policy matches the existing helper.
State dictionaries preserve data/configuration, not train/eval mode or optimizer
lifecycle. Save/load the optimizer separately; use `weights_only=True` for the
plain state payload.

Bounds remain dimensions<=4096, weight elements<=4M, activation/output<=1M,
MAC<=128M, positive row count<=4096. These are operand/work caps, not a guarantee
of total allocator/process memory. No float base is retained/materialized by the
module; checkpoint serialization overhead and overall peak memory are unmeasured.

## Qualification

295 checks across ten guarded ordinary-launch gates on `.104`, daily NCCL-off
and explicit `rccl-single` cohorts:

| Checks per cohort | Evidence |
| ---: | --- |
|32 module | Three dtypes × three bias policies × two input layouts; independent scalar CPU packing/CPU-double output/input and bias-gradient references; source independence/guards; dtype/device preservation; explicit AMP; ten refusals. |
|36 state | Nine CPU `weights_only=True` file/factory/direct/nested roundtrips;24 malformed payload refusals with own-module state unchanged, including deliberately changed valid weights before each invalid field; partial/assign/oversized-factory refusals. |
|3 optimizer | Two owned FP32 rank2 adapter steps with independent CPU base/first-gradient/AdamW references; CPU checkpoint after step1 and resumed step2 exactly match continuous GPU model/optimizer tensors. |
|71 retained | Existing recipe: layouts/zero rows/ties/all INT8 bytes/private stream/finite differences/adapter/AMP/higher-gradient policy/25 refusals/native GPU trace. |

Daily5 and opt-in6 consumer checks additionally retain normal runtime identities
and launch behavior. Fixtures use no user application/model/dataset. These are
bounded eager first-derivative and two-step resume results, not model convergence,
quantization quality, long-run stability, graphs/compiled W8, full QLoRA or speed
qualification. Native direct weight-only backward/packed INT4 refusals remain.
TorchAO/bitsandbytes/NF4 integrations remain absent/unqualified. Historical Python
CPU segfault remains unresolved and distinct from fixed VM-heap/CSR/hostcall defects.

No native build, ccache operation, image/package installation, active-prefix update,
Torch/ROCm/kernel/default change, deletion or `.105` contact occurs. New optional
helper/test files are staged in `WS/helpers`; existing builds/caches/source/history,
links/boot fallbacks and user applications are preserved. Tests use the ordinary
shared runtime lock; no exclusive update is needed because no active prefix is
changed. Existing math SDPA/SDMA0/heap-off/N3.2 settings remain.

Raw report: `WS/logs/quant-b055-qualification-r1/final.json`; per-gate result,
exit, kernel and guard logs are retained. Local archive/checksum details are in
[archive.json](archive.json); source/runtime/boot/protected-path identities are in
[verification.json](verification.json).29 local BC250 tests and selected Ruff
format/error checks pass.445 protected hashes, four AOT DBs and compatibility
links remain unchanged. Local archive510,746B SHA256
`2c0099691e7744e37f4623a269df593d90a48009731bd56de969bc5cbc1b57e7`
contains96 raw log/fixture files (2,070,943B), six verified tools and ten guard
records. Same r2 boot`c6a50cfa-a368-4ba5-ab28-66250b4f1f1f`; daily runtime561f014e,
RCCL profile446f345e and native Torch sourcea5249802 retained.
Ten guards acknowledged; kernel logs unchanged per gate
and through the batch; no own jobs/containers/armed guards remain at handoff.

**Next B05.6:** package the explicit W8 helper/module in an independent optional
runtime candidate, then qualify public imports, state/gradient contracts and
retained consumers before any locked promotion. Preserve dense defaults and old
images; no automatic conversion or TorchAO/full-QLoRA admission.
