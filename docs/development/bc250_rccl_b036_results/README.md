# B03.6: isolated Torch/RCCL public contracts and broader regression

The unchanged isolated B03.5 Torch/RCCL candidate passes **1,365 checks across
21 gates**, including178 new public-contract checks. No production defect was
reproduced, and no Torch/RCCL/native library build, ccache operation or active
runtime update was needed. Daily Torch still has NCCL disabled.
[Exact verification and identities](verification.json).

Tooling revision:`9dc76a795`.

## Selected public contracts

The owned fixture runs on native gfx1013, one physical GPU/one RCCL rank. Its178
checks comprise:

- 128 low-level identity operations:8 dtypes × sizes17/257 × disjoint output/exact
  alias × AllReduce/Reduce/AllGather/ReduceScatter. Offset5 inputs/outputs,
  prefix/suffix guards, unchanged inputs and immediate private-stream consumption
  compare exactly against CPU values. Eight broadcast identities are separate.
- Five zero-count collective wrappers, four device/layout availability predicates
  and12 low-level host refusals: sequence/length/CPU/stride/dtype/size/duplicate
  device/stream length/root/int16/complex. Error text is checked; no bad device
  pointer or partial-overlap experiment is submitted.
- Three init/PreMulSum/destroy cycles (six composites), subgroup PreMulSum/destroy,
  and cross-stream event handoff/immediate consumption. Nonzero FP32 quarter
  values multiplied by2 supply actual GPU arithmetic references.
- Eight process-group refusals:CPU/stride/bitwise AND-OR-XOR/bool AVG/two fnuz
  FP8 formats. These are host validation checks, not FP8 arithmetic qualification.
- Four DDP composites:BF16/FP32 × bucket views off/on, two SGD updates, each with
  two microbatches and `no_sync` on the first. Identity inputs/constant1/8 weights
  give independent exact CPU-float64 output/gradient/update references (gradient
  4*W and update0.75*W). This deliberately small analytical fixture does not
  qualify arbitrary BF16 accumulation, other optimizers or long training.
- One actual loaded isolated DSO check.

The eight low-level types are FP16/BF16/FP32/FP64/int8/uint8/int32/int64. Rank-one
ordinary collectives are identities:these do not establish communication,
Ring/Tree/LL runtime, multi-device/P2P/XGMI/GDA/SDMA or scaling correctness. Exact
alias is covered; partial overlap, concurrency, graphs, allocator registration,
communicator offload/shrink and higher gradients remain unqualified. No speed or
memory improvement is claimed.

## Broader regression and applicability boundary

The retained B03.5 RCCL64 and GPU Gloo41 checks pass, followed by the existing
18-gate stack suite (1,082 checks):CPU Gloo/store31, sparse253/triangular80/
legacy41, FFT245/real72/contracts3, vision51/integration9, audio34/native40,
images86/video55/files46, compiled dense14/CNN14, profiler1 and core7. CPU Gloo
uses its existing explicit `--nccl-available` option; numerical checks and
tolerances are unchanged. GPU Gloo remains host-staged on two local ranks sharing
one physical GPU. No user application is inspected or used.

Source inspection precedes testing. [Reviewed source identities](contract-audit.json)
show the low-level availability predicate checks device/layout uniqueness, not
complete dtype/hardware eligibility. FP8 storage maps to uint8 in the low-level
wrapper; that mapping does not qualify numeric FP8 reductions. Process-group
fnuz formats are explicitly refused; version-admitted e4m3fn/e5m2 and filtered
symmetric/registration/offload paths need a separate applicability/guard audit.
The B03.4 RCCL remains a375-object filtered build, not a full RCCL distribution.
The candidate therefore remains isolated even after these broader regressions.

## Reproduction and preservation

On .104, `WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Invoke helpers through the
existing exclusive `WS/helpers/update_dev_prefix.py -- ...` lock:

1. Stage the B03.6 fixture/qualifier/exporter and the shared qualifier's pending
   accepted-ID update as `helpers/qualify_training_n33.rccl-b036-next.py`.
2. `setup_rccl_b036_on_host.py` backs up/atomically installs the shared helper.
3. `qualify_rccl_b036_on_host.py --attempt r1` uses the existing
   `WS/rccl-b035/candidate-release-r3`; it does not create/rebuild/deploy a release.
4. `export_rccl_b036_on_host.py` exports the accepted evidence once.

Final preservation baseline:`WS/logs/rccl-b036-qualification-r1/final.json`.
All340 protected hashes, four AOT databases and links are unchanged. All21 guards
acknowledged; accepted gate and whole-batch kernel logs match. Helper backup:
`WS/recovery/rccl-b036-before`. Raw evidence and verification are local in
`kernel/artifacts/rccl-b036`; native payloads remain in the already verified
B03.5/B03.4 artifacts, with no redundant binary export. Archive6,543,360B matches
SHA256d083ff68a56bee95053784c7943da4cfcd4c63c7000c0a5880af6ab5cb49e31b;257 raw
logs total5,386,517B. Active Torch2b449c4/
image506f3b26/runtimeJSON32fd4559/native HIPa1162308/systems1f9609 and safe defaults
remain unchanged. Candidate Torchade6fcc/native libtorch_hip7f9e69a4 and
RCCL38fd6/DSOca790dc7 also remain unchanged.

Same r2 boot2db1c7c2; no reboot/.105/BIOS/fallback/user application touch/deletion.
Historical unknown Python CPU faults remain unresolved and distinct from fixed
VM-heap/CSR/B04.2 hostcall queue defects. No own jobs/root SSH remain at handoff.

**Next B03.7:** audit and guard the isolated candidate's excluded/version-admitted
public paths (especially FP8 reduction versus byte-copy semantics and symmetric/
registration/offload APIs), then determine the bounded single-device promotion
profile. Keep the daily prefix unchanged until those boundaries are concrete;
one physical GPU cannot qualify a multi-device backend.
