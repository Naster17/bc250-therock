# N2.18 scalar CK DL grouped convolution — 2026-10-19

TheRock code `746528fa6`. Existing installed CK headers pass **128 FP32 GPU
convolution CPU-reference comparisons and eight argument refusals**. Retained
FP32/FP16/signed-INT8 GEMM examples and seven public Torch core regression groups
also pass. No production CK arithmetic defect was reproduced; no operator,
Torch routing, runtime library or performance change is claimed.

## Audit and tested contract

Patch0005 already adds gfx1013 to `is_gfx101_supported()` and admits scalar DL
classes. Earlier `validate_ml_on_host.sh` runs verified only the three GEMM
examples. Architecture admission alone was not convolution qualification.
The selected `DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK` calls scalar
`GridwiseGemmDlMultipleD_km_kn_mn`; the executable checks native gfx1013 admission
and `!is_xdl_supported()`. No XDL/WMMA or modern MIOpen CK plugin is enabled.

The new consumer uses the upstream FP32 example's 256-thread/128×128 scalar
configuration, with explicit `G_NHW_C/G_K_YX_C/G_NHW_K` strided layout tags.
These flatten spatial dimensions: the padded cases use four unused channels
between pixels/filter positions and16 elements between groups; spatial rows and
batches remain linear. They do not promise arbitrary independent row strides.

| Shape N/H/W/C/K/filter | Stride / dilation / left,right padding |
| ---------------------- | -------------------------------------- |
| 1/5/7/4/4/1            | 1 / 1 / 0,0                            |
| 2/7/9/8/12/3           | 1 / 1 / 1,1                            |
| 1/9/10/4/132/3         | 2 / 1 / 1,0                            |
| 1/8/9/8/8/3            | 1 / 2 / 2,1                            |

Four shapes × groups1/3 × contiguous/padded × dyadic/non-dyadic inputs ×
plain/Add+ReLU epilogues × default/private HIP streams =128 numerical cases.
Independent scalar CPU double accumulation uses actual FP32 input values;
dyadic outputs must match exactly, other outputs use fixed
`2e-5 + 5e-5*abs(reference)` tolerance. Maximum absolute error4.255575e-6.
256-byte guards, output stride holes and bytewise immutable input/weight/D
storage pass. Eight C=3/K=5 vector-incompatible arguments are refused without
launch and leave storage unchanged. Initialization/verification synchronize;
this is not overlapping-stream or asynchronous-completion qualification.

## Build, failures and recovery

Only one standalone consumer is compiled per attempt using ccache20GiB, OMP4,
one compiler job, `-O2 -std=c++17 --offload-arch=gfx1013`. Builder image
`fa3e8c254b6a791b541d49d4b0a1000a5d18aaa661b41e89794b82c104770238`, read-only
source/build/helper mounts, writable output/cache, no GPU/network during build.
All four attempt directories and caches are retained.

- r1: test-only compile error, incorrectly qualified global `StreamConfig` with
  `ck::`; corrected. No GPU execution.
- r2: build passes, existing launcher cannot arm its r3-default guard on r2.
  No GPU execution. Added explicit `--guard-release` restricted to the retained
  r2/r3 releases; existing r3 default and original recovery guard unchanged.
  Kernel mismatch now fails before launch; early guard exit reports its log.
  Old helper retained at `WS/recovery/ck-dl-n218-before/`.
- r3: packed cases pass; the initial padded fixture wrongly supplies arbitrary
  row/batch gaps under packed layout tags. Corrected test layout/storage contract,
  without changing CPU reference or tolerance. No production header patch.
- r4: all136 direct checks, three retained GEMM examples and core7 pass.
  The r2 kernel with the launcher's unchanged r3 default is separately refused
  before a guard/container is started: [negative check](default-mismatch.log).

The board was already running **7.1.5-bc250rocm-r2** on entry,
boot `d0532075-88fb-4a2f-974e-f61578d20d9f`; no reboot or boot-policy change was
made. Prior r3 qualification remains historical evidence, not this batch's
kernel. Hardware watchdog active; failed r3 GPU fixture and three final GPU
processes all have acknowledged600s guards. Four attempts' dmesg is unchanged.
191 protected identities match, including earlier binaries, CK headers, DSOs,
links/runtime metadata and four persistent FFT AOT databases. New binary SHA
and eight helper hashes are in [final preservation](ck-dl-n218-qualification-r4/final.json).
[Attempt/guard summaries](gate-summary.json) retain failures and kernel hashes.
No running agent job/container/guard; free about7.16GiB at qualification end.

Exact replay, under the existing exclusive runtime lock:

```sh
python3 WS/helpers/update_dev_prefix.py -- \
  python3 WS/helpers/qualify_ck_dl_n218_on_host.py --attempt r5
```

Use a fresh attempt number. Full local ignored copies, including binaries and
failed sources: `kernel/artifacts/ck-dl-n218/`; remote logs `WS/logs/ck-dl-n218-*`,
builds `WS/build/bc250-coverage/ck-dl-n218-r{1,2,3,4}`. Next preservation baseline:
`WS/logs/ck-dl-n218-qualification-r4/final.json` plus its `binary_sha256`.
Installed Torch `dc8fea0`, source tests `2b449c4`, native libraries `319a235d6`,
source-doc `c29021b40e7` unchanged. Historical CPU Python fault remains unresolved
and separate from fixed VM-heap/CSR corruption and the test-only failures here.

Limits: forward only, FP32 and this scalar configuration/layout/epilogue scope.
No convolution backward, FP16/INT8 convolution, arbitrary layouts, special-value
policy, large-index/overflow, graph/concurrency, tuning or performance claim.
No automatic Torch/MIOpen integration follows from a direct CK invocation.

Next: review remaining N2 gaps by practical missing capability/defect, then
select the next bounded N3 training/functional batch. Broad N2 remains open;
there is no automatically required N2.19 or endless API-combination sweep.
