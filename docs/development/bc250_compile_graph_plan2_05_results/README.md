# PLAN2-05 — torch.compile expansion + graph-capture integration qualification

**PASSED 2026-11-14** on .104 (release `10.2-gfx1013-20261111-138a66f52`,
Torch `2.13.0a0+gfx1013`, boot `0d46a6f1` unchanged). Fifth PLAN2 item.
Scope (user-confirmed): **qualify on OWNED fixtures** — expand compiled-training
coverage and qualify HIP graph capture of a LoRA-pattern training step, with a
reusable opt-in capture helper. **No app entry-point, Torch source, default,
package or release change** (pytorch series stays 23 patches); not installed
into any image.

## Verdict

- **Part A — compile expansion: 14/14 checks** (cg-r5 attention+LoRA 7/7,
  cg-r7 embed_ce+dynamic 7/7). `torch.compile(fullgraph=True)` with real Inductor
  kernels (3–11 per fixture, no eager fallback), parity vs an independent CPU
  reference and vs eager.
- **Part B — graph integration: 14/14 checks** (cg-r12 in-process 10/10,
  cg-r13 fresh-process resume 4/4). **Every graph parity case is bitwise-exact
  (rel-L2 0.0)**: graphed fwd+bwd + eager AdamW (FP32/FP16/BF16 autocast),
  whole-step capturable AdamW, gradient checkpointing, `make_graphed_callables`,
  and a 16-update fresh-process recapture/resume.
- **Speed:** graph capture of the owned launch-bound LoRA stack gives
  **1.92×** whole-step (16.31→8.48 ms; 1.9–3.3× across runs). torch.compile does
  **not** speed up these tiny fixtures (attention ~1.0×, LoRA/embed 0.61–0.68× —
  compile overhead dominates at small geometry). Consistent with D70: the real
  LoRA step is GEMM-compute-bound, so graphs buy the ~10% non-GEMM launch share,
  not the GEMMs.

## Part A — compile expansion (correctness coverage, not a speed claim)

Owned fixtures: causal math-SDPA attention block, LoRA linear (frozen base +
rank-8 adapter), embedding+LM-head+cross-entropy, dynamic-shape attention.
FP32/FP16/BF16; `fullgraph=True`; Inductor `generated_kernel_count > 0` required.

Parity (worst rel-L2): FP32 vs CPU-FP64 ~1e-6–2e-7; compiled-vs-eager ≤3e-5
(FP16 attention), LoRA/embed essentially bitwise. Frozen-base grads absent;
trainable grads finite/nonzero. Dynamic fixture (FP32) reuses one compiled graph
across L128/192/256 with tight parity.

Two precision findings (recorded, attributed to dtype/implementation — NOT
compile defects; compiled-vs-eager stays tight):
- **FP16 attention backward** deviates ~3.4% from a matching-dtype CPU reference
  and ~24% from CPU-FP64 — and EAGER deviates identically. FP16's narrow exponent
  range in the softmax backward (BF16, wider exponent, deviates only ~0.4%).
- **Dynamic FP16 compile** lowers SDPA to a fused Inductor kernel differing from
  eager math SDPA by ~0.7–0.9% (L192/L256; first specialization L128 tight). The
  gated dynamic fixture therefore runs FP32 to isolate shape reuse.

Compile contract applied: compiled matches eager within the tight eager bound,
AND compile adds no error beyond eager relative to the independent CPU reference
(CPU-anchor gate = max(bound, eager's own deviation)). Numeric thresholds were
never relaxed; the methodology refinements are recorded in BOUNDS.md.

## Part B — graph-capture integration (bitwise-exact)

Owned LoRA-pattern stack: 24 frozen-base + trainable-LoRA residual blocks,
width 128, AdamW, optional non-reentrant checkpointing. Capture via the opt-in
`GraphedStep` helper (side-stream warmup + `torch.cuda.graph`) and
`torch.cuda.make_graphed_callables`.

Multi-update (8) parity vs eager from identical init — **all rel-L2 0.0**:
loss trajectory, adapter weights, optimizer exp_avg/exp_avg_sq, for graphed
fwd+bwd + eager AdamW (FP32/FP16/BF16 autocast with FP32 masters), whole-step
capturable AdamW, and checkpointing + graphed fwd+bwd. `make_graphed_callables`
forward+backward bitwise. **Fresh-process recapture/resume**: graphed 8 updates
→ save → FRESH process loads + recaptures + 8 more; the combined 16-update
trajectory + final weights + optimizer state match a single-process eager
16-update reference **bitwise (0.0)**.

Refusals verified: dropout>0 capture refused (RNG not graph-safe); a static
input carrying a stale eager autograd graph refused. Documented boundaries:
static shapes only (new shape → recapture); the graph memory pool is not
checkpointable, so resume requires a fresh recapture from loaded state (proven
exact above).

Whole-step timing (qualified green run cg-r12): eager 16.31 ms vs graphed
8.48 ms = **1.92×** (other runs 2.58/3.04/3.32×; the owned fixture is
launch-bound). On the real LoRA model the gain is bounded by the ~10% non-GEMM
share (D70) — no general whole-model speed claim.

## Reusable opt-in helper

`build_tools/bc250/compile_graph_plan2_05/graph_step.py`: `GraphedStep(step_fn)`
captures a zero-argument step closure over static tensors and replays it with one
launch; `assert_graph_safe` guards dropout/stale-graph; `capture_graphed_callable`
wraps `make_graphed_callables`. Import deliberately; changes no default. Recipe:
keep parameters/inputs/optimizer-state static, warm up on a side stream, drop any
eager output graph before capture, zero_grad(set_to_none=False) inside the
closure; for whole-step capture use `AdamW(capturable=True, foreach=False)` and
reset weights/state to init after warmup; run the optimizer eagerly outside the
graph for the fwd+bwd-only variant.

## Attempt journey (harness fixes, not stack defects; all retained)

cg-r1 frozen-base diff-target bug (autograd.grad over non-requires-grad params);
cg-r2/r4 FP16 attention CPU-anchor attribution (eager deviates identically →
"no worse than eager" rule); cg-r3 stale `del cpu_model`; cg-r6 dynamic FP16
fused-lowering finding → FP32 gated dynamic; cg-r8 `opt.state` keyed by parameter
tensors not ids; cg-r9 whole-step warmup advanced weights/state → reset to init
after capture; cg-r10/r11 whole-step reference matched to `capturable=True,
foreach=False` (capturable-vs-not differs ~1e-5 in optimizer state) → bitwise.
No numeric bound, drift gate or envelope parameter was relaxed. 16 harness tests,
suite 467, Black clean.

## Envelope and safety

Every run under the qualified PLAN2-01/02/03/04 guardian (SH lock, nonce-acked
600 s boot-guard, 460 s child, `timeout 430` backstop, kernel_noise filter,
registry identity, foreign-container refusal) + stock-clock gate ≤2000 MHz.
Boot `0d46a6f1` unchanged across all runs; kernel logs identical (0 benign
added); registry unchanged; 0 foreign containers; boards idle pre/post; .105
untouched; host governor/clock/watchdog/kernel NOT modified. Owned synthetic
fixtures only; no user application; transformers (unused here) would come from
the owned lora-smoke env. No default/package/release/source change.

## Bounds and remaining scope

Compile: small owned geometries, FP32/FP16/BF16, fullgraph; not a throughput
claim; dynamic FP16 fused-attention lowering and FP16 attention backward
precision are recorded properties. Graphs: static shapes, dropout=0, eager
first-order; resume requires recapture; whole-step needs capturable optimizer.
Real-model integration (wiring graphs into a training entry point) and
compiled-optimizer/large-model throughput are separate future scopes.

## Reproduction

```bash
python3 build_tools/bc250/compile_graph_plan2_05/run_compile_graph.py --label cg-r5 --script compile --part 1
python3 build_tools/bc250/compile_graph_plan2_05/run_compile_graph.py --label cg-r7 --script compile --part 2
python3 build_tools/bc250/compile_graph_plan2_05/run_compile_graph.py --label cg-r12 --script graph --part 1
python3 build_tools/bc250/compile_graph_plan2_05/run_compile_graph.py --label cg-r13 --script resume --part 1
```

## Evidence

`kernel/artifacts/compile-graph-plan2-05/` (92 files): BOUNDS.md (+addenda),
compile cg-r1–r7, graph cg-r8–r12, resume cg-r13, guardians/preflights/kernel
logs for every run, orchestrator logs. Host task dir
`/home/nik/bc250work/compile-graph-plan2-05` retained. Publication:
bc250-therock tools+report (no source change), umbrella curated PLAN2-05.