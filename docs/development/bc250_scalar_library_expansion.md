# BC-250 scalar library expansion

Implemented locally and GPU-tested on `.104`; the frozen daily-use release and
fallback remain unchanged. `.105`, kernel, SDMA policy and existing data are preserved.

| Component             | GPU evidence                                                                                                                                                  | Current limits                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| hipBLASLt             | 403 C API cases: FP32/FP16/BF16, layouts/transposes/batches, host/device scalars, bias, ReLU and GELU                                                         | No grouped/pointer-array/C++ extension backend, auxiliary/backward epilogues or tuning                                   |
| hipTensor contraction | 76 cases: four homogeneous real types plus FP16/BF16→FP32 and FP32→FP64, strides/modes/batching, alpha/beta, unary ops and rejection cases                    | Seven explicit type combinations, 16 modes, fixed algorithm, zero workspace; no complex tensors or unilateral reductions |
| hipSPARSELt           | 162 cases: strip prune/check/compress/matmul; genuine maximum-L1 FP16/BF16 4×4 TILE pruning; independently decoded compression and FP64 CPU matmul references | No FP32 TILE, integer types, fusion, split-K or multiple streams; backend-specific compression                           |
| CK Tile               | 19 CPU-reference runs: reduction, permutation, padded transpose, strided elementwise ops, 3D max pooling/indices and FP16/BF16/FP32 RMSNorm                   | Selected generic paths; matrix engines remain disabled                                                                   |
| hipTensor generic     | 23 additional rank-three/four reduction, permutation, binary and trinary cases                                                                                | Wider ranks/operators/layouts remain                                                                                     |
| hipDNN                | 14 CPU-reference LayerNorm/RMSNorm backward and batchnorm training/backward graph cases                                                                       | Wider graphs, bundles and provider coverage remain                                                                       |
| RPP                   | 16 image/layout cases plus 26 host/GPU audio cases, including 1D/2D decibels, automatic/non-unit references, all-zero inputs and padding guards               | Wider types/image/audio/volume operations remain                                                                         |
| Torch                 | 55 broader CPU-reference operator checks (~40 seconds), plus Inductor, LoRA updates and checkpoint roundtrip                                                  | No new long stability qualification or performance tuning                                                                |

Key fixes: native CK Tile registry/launch geometry and stride handling, Torch
channels-last pooling gradients, RPP decibel reduction/reference handling, sparse
Search synchronization and zero-scalar semantics, and colliding extracted kernel
packs. hipSPARSELt now has a separate `sparselt` artifact; consumers need it alongside
`sparse`. Its scalar HIP source is explicitly tracked for incremental rebuilds.

Build helpers: `build_scalar_backends.sh` and `build_ck_tile_coverage.sh`, using
16 ordinary jobs, 12 hipTensor jobs and the existing 20 GiB ccache. The scalar
Lt profiles are opt-in: `THEROCK_GFX1013_SCALAR_BACKENDS=ON` and
`THEROCK_ENABLE_SPARSELT=ON`, restricted to the exclusive gfx1013 family.

Bounded GPU runner: `validate_library_expansion.py WORKSPACE --group scalar`
(other groups: `ck`, `tensor`, `dnn`, `rpp`). Reports are in
`.104`'s `logs/library-expansion/` and the adjacent committed results directory.
Audio evidence is in `logs/rpp-audio-validation.log`; all 26 cases passed.

Candidate image: `localhost/bc250-stack:10.2-gfx1013-scalar-candidate`.
The new Torch source is `b36877506d38eb74abcd01d667e9b05d6b60430a` (editable pip);
ordered patches are retained under `patches/bc250-pytorch/` and the existing
`patches/amd-mainline/rocm-libraries/` chain. Sources, objects, caches, images,
containers, checkpoints and previous releases are retained.

The first full Torch run overlapped a developer-prefix rebuild and suffered a
host segfault. Isolated sparse checks and the full run after the build passed;
its precise cause is unconfirmed, and failure logs are preserved. Run GPU gates
only after the shared prefix build completes. This candidate is not an activated
or newly sealed daily-use release; other-architecture GPU runs remain unqualified.
