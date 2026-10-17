# A06: native gated-delta preparation — 2026-10-17

This milestone implements the first bounded native gfx1013 gated-delta substage:
Q/K normalization, query scaling, beta gating of K/V, padding/layout conversion
and per-chunk cumulative log-decay, with Q/K/V/log-decay/beta first-order backward.
The explicit `torch.nn.bc250_gated_delta` module also composes this preparation
with ATen triangular solves, dense GPU matmuls and a sequential GPU chunk scan.
It is not the FLA package and does not replace all gated-delta computation with
fused native kernels. The existing LoRA trainer/helper is unchanged.

## API

```python
from torch.nn.bc250_gated_delta import prepare_gated_delta, chunk_gated_delta_rule

# q/k: [B,L,H,Dk], v: [B,L,H,Dv], g/beta: [B,L,H], on actual gfx1013.
# Expand grouped key heads before calling, as Qwen3.5 already does.
prepared = prepare_gated_delta(q, k, v, g, beta, normalize=True)
out, state = chunk_gated_delta_rule(
    q, k, v, g, beta,
    use_qk_l2norm_in_kernel=True,
    output_final_state=True,
)
```

Q/K/V share FP32, FP16 or BF16 dtype; both gates independently accept these
types. The prepared outputs are FP32 `[B,H,ceil(L/64),64,D]` Q/sqrt(Dk), K,
beta*K, beta*V, plus `[B,H,ceil(L/64),64]` cumulative log-decay. Q/K normalization
uses eps1e-6. Value padding is zero; cumulative decay retains the last real sum
through padding. Output padding gradients correctly contribute to real decay
gradients. Output from the complete operation has query dtype and BLHV layout;
optional final/initial state is FP32 `[B,H,Dk,Dv]`. State gradients are supported
in the tested first-order contract. Source inputs and initial state are not mutated.

Chunk size is64; head widths1–128, nonempty tensors, and at most1,048,576 padded
B*H*L entries. Strided input copies remain GPU operations with autograd edges.
All tensors must share one actual gfx1013 device; no architecture override is
used. Gates follow the reference's expected nonpositive log-decay convention.
Finite ordinary inputs are the tested domain; this is not extreme-conditioning
qualification. Initial-state and empty/shape/dtype/architecture refusals are
explicit. Unsupported inputs should retain the original application backend.

Use explicit imports/calls. No model/Transformers hook, package replacement,
warning filter or automatic sparse/SDPA selection is installed. The ordinary
user LoRA command therefore retains its existing gated-delta reference warning.
Packed/variable-length sequences, decoding/cache integration, higher-order or
forward AD, compile/export/graphs and sustained convergence are unqualified.

## Validation and measurement

The tests use native gfx1013 in the qualified rootful runtime on .104, protected
by the exclusive prefix-update lock and existing600s recovery guard. Independent
CPU float64 references check the preparation outputs/gradients and a separate
token-by-token recurrence, rather than only comparing two chunked implementations.
The recurrence includes both gate gradients, Q/K/V and initial/final state.
Non-power-of-two widths, unequal key/value widths, partial chunks, multiple heads/
batches, strided inputs, custom streams, unused output gradients and zero norms/
gates are covered. Tolerances were fixed before tests; none was relaxed after a
numerical failure.

Model tests use synthetic random tokens and the existing causal-conv/chunked-loss
helper identically on both paths, with math SDPA. They never read user datasets
or existing checkpoints. Small random Qwen3.5 LoRA tests cover FP32/BF16 with and
without checkpointing. Real downloaded Qwen3.5-2B BF16/rank16 uses checkpointing,
65 and1024 tokens and all372 adapter gradients. The per-operation CPU oracle is
separate from model GPU reference comparisons. No optimizer/convergence/resume
claim is inferred from a forward/backward timing experiment.

Paired operator timings use six alternating rounds, preparation x5/full x1,
after two warmups. They time forward+autograd.grad with HIP events and wall time,
including host submission. Model timing uses three alternating pairs after both
paths warm up; it includes forward/backward and excludes CPU gradient copies.
Peak allocations are measured per call; empty_cache is outside model timing.
These bounded fixtures are not a universal/application-training speed guarantee.

The initial model adapter refused forwarded Transformers `use_cache`,
`output_attentions` and `output_hidden_states` arguments in three successive
runs before the test adapter explicitly handled these outer-model flags.
Those failures and unchanged kernel logs are retained; the production API was
not broadened to silently discard unknown arguments.

## Provenance and recovery

Full sources remain in the patched local Torch repository. The incremental
patch/bundle is based on6ed3d12; the older three-file SDPA patch-reconstruction
gap remains separate. No C++/ROCm/kernel build or image change is required for
this Python/Triton module; existing native Triton JIT/cache is reused, ccache is
untouched. All active shared libraries, MIOpen databases, previous/frozen trees,
shared consumers and original trainer files remain protected. .105, BIOS, boot
fallback, SDMA0/experimental-heap-off and default dense/math policies are unchanged.
The historical Python CPU segfault remains unresolved and distinct from the
fixed VM-heap/CSR corruption.

## Installed results

Torch revision: `ba6cbcd2fafac1a6503f1f580033fbf72ad2a4f9`.
TheRock runtime helpers: `e61d609d3658ee984907841f4fe1dc8ecdf66baa`.
Module SHA256: `9ecb6f9014bc139e54cffe0fbb2c80f782035e57a3149e3c9a6b31fe445e8b2b`.

| Gate | Result |
| --- | --- |
| Candidate / installed API |40 each:18 preparation,18 recurrent/state,2 exact64/low-precision-gate,1 five-output-subset/zero group,1 eight-refusal group |
| Small random Qwen3.5 LoRA |4 pairs, FP32/BF16 with/without checkpointing; max gradient relativeL2~4.52e-7 FP32 /0 BF16 |
| Actual Qwen3.5-2B LoRA |65/1024 tokens, BF16/rank16/alpha32/all decoder linear targets/checkpointing;372 gradients each; max relativeL2~3.23% /3.21% |
| Existing installed regressions |69 A05 training tests and seven core groups pass |
| Guard/kernel evidence |All12 guards ACK/600s and byte-identical dmesg53f83f7b; three retained harness refusals are not passing tests |
| Compiler audit |210 cached candidate/installed variants, gfx1013/wave32/four warps/no MFMA/WMMA; metadata/instruction audit, not a runtime trace |
| Lint |Nine configured static/format checks through spin, plus spin quicklint invoking lintrunner -a; no full-project typing/CI claim |
| Protection |41 hashes unchanged, including active DSOs, prior/frozen cohorts, ROCm/DB/kpack, shared consumers and original trainer/helper |

Installed operator medians, forward plus all five input gradients:

| Type / length | Reference prep ms | Native prep ms | Reference complete ms | Native complete ms |
| --- | ---: | ---: | ---: | ---: |
| FP32 /65 |1.333 |0.772 |6.148 |5.465 |
| FP32 /1024 |2.690 |2.006 |27.760 |25.722 |
| BF16 /65 |1.442 |0.769 |6.572 |5.903 |
| BF16 /1024 |2.893 |1.991 |26.628 |25.995 |

Geometry is B1/H16/Dk128/Dv128. Small-call host submission and creation of output
gradients are included equally on both paths. The speed ratios vary by fixture;
the remaining ATen solve/scan dominates complete-operator time.

Actual-model synthetic forward/backward medians:

| Length | Reference s | Native s | Ratio | Reference peak bytes | Native peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
|65 |0.627804 |0.619325 |1.01369 |4,652,103,168 |4,627,975,680 |
|1024 |7.653459 |7.390244 |1.03562 |11,976,548,352 |11,745,457,152 |

BF16 model outputs/gradients are not bitwise equal.65-token losses were
15.100475/15.096644;1024-token losses14.304518/14.304263. The maximum adapter
relativeL2 limit0.04 and loss-difference limit0.015 were set before these runs;
independent CPU operator references use tighter type-specific tolerances in the
test source. These checks do not establish long-run optimizer/convergence parity.
No dataset throughput or checkpoint update was measured.

Evidence: [API](api-installed.json), [operator samples](operator-timings.json),
[small model](model-tiny.json), [actual model](model-actual.json),
[guards](gates.json), [compiler summary](compiler.json),
[lint](lint-final.log), [deployment/protected hashes](deployment.json).
Raw commands/logs remain at `WS/logs/gated-delta-a06-*` and the local ignored
`kernel/artifacts/gated-delta-a06/remote-logs/`. A06 source/bundle is retained
locally and in `WS/helpers/torch-a06.bundle`; patch0006 reverse-apply and bundle
prerequisites verify against full source6ed3d12. The report's original operator
baseline (`gated-delta-a06-profile-r1`) also remains in those raw logs.

Installation added one module to the existing active `WS/pytorch-sparse-followup`
under the exclusive runtime lock. It changed no shared library or existing module.
`SOURCE_GATED_DELTA_A06.json` in WS/current records revision, hashes and gates;
`runtime.json`/`GIT_REVISION` identify the new source. Recovery metadata is at
`WS/recovery/gated-delta-a06-before/`; rollback restores that runtime.json and
GIT_REVISION and removes only the added module under the same exclusive lock.
No agent build/test/container or guard deadline remains. Disk8.4GiB free.

A06's recorded first-substage milestone is complete. Full native gated-delta
solve/scan and automatic model integration remain future work. Next primary
item is A07: inspect the existing tiled SDPA head128 bound, extend to256 with
bounded output/QKV-gradient, GQA/mask/all-masked and LoRA references, keep math
default; dropout/compile/graphs are separate contracts. A07–A09 precede N2.7.
