# PLAN-09 — modern model/environment compatibility (D62 + D63)

Completed 2026-11-12. PLAN-09 makes the supported package combinations explicit
and delivers a **reusable owned new-model admission fixture**. Both deliverables
are done: the [compatibility matrix](compatibility-matrix.md) and the admission
fixture `build_tools/bc250/admit_model_d62.py`, qualified by a bounded GPU run on
the .104 release.

## Deliverable 1 — compatibility matrix

[compatibility-matrix.md](compatibility-matrix.md) records the **three** distinct
consumer environments riding native Torch 2.13.0a0+gfx1013, with versions verified
on-host (not just documented):

| Environment | transformers | accelerate | peft | other |
| --- | --- | --- | --- | --- |
| Frozen consumers (`bc250-train`) | 5.2.0 | 1.12.0 | 0.18.1 | triton 3.8.0, numpy 2.5.1, hub 1.4.1, safetensors 0.8.0, tokenizers 0.22.2 |
| lora-smoke app (L01/L02) | 5.22.1 | 1.15.0 | 0.21.2 | trl 1.17.1, datasets 5.1.0 |
| Qwen3.5 native op overlay (A06/A07) | 5.19.0 | — | — | gated to `qwen3_5_text` |

Frozen versions read from the release `python-consumers/packages.json`; lora-smoke
from its venv dist-info + `python-origin.json`. The matrix also records consumer
wheels, qualified vs absent kernel routes (W8 frozen; NF4/packed-INT4/full-QLoRA
absent), and requalification rules.

## Deliverable 2 — reusable owned admission fixture

`admit_model_d62.py` admits a model architecture through a bounded battery and
records the actual kernel routes:

- **Owned-only**: config-driven registry (`gpt2-tiny` = the D26 21280-param
  identity, `qwen3-tiny`, `qwen35-hybrid-tiny` lifted from the excluded module's
  random-init branch). Random init from config; no downloaded weights, no
  `from_pretrained`, and an import guard that refuses any `finetune-n7` sys.path.
- **Battery**: load → GPU forward vs an independent CPU/FP64 reference →
  finite/nonzero backward → optimizer step → **bitwise** checkpoint save/resume →
  exact resumed continuation (D59 backbone; save precedes the reference
  continuation; raw-byte fingerprint works for fp32/bf16/fp16).
- **Route probes** (op-level, owned): attention backend flags (efficient/flash
  must refuse, quiet math fallback), native tiled-SDPA, gated-delta, causal-conv,
  frozen-W8 — each recorded fired / refused / absent. Failures are recorded in the
  report, never crash it.
- Pure logic is importable without torch (workstation-testable); the GPU battery
  uses lazy imports and runs under `bc250-train` on the board.

## D63 bounded GPU admission (.104 release)

Five representative owned combinations, all **passed**, all four routes fired,
board clean (0 containers before/after). Fixture sha256 `28414a0c…` matches every
report. Evidence: `kernel/artifacts/plan09-d63/`.

| Model | dtype / backend | Forward rel. err | Resume |
| --- | --- | --- | --- |
| gpt2-tiny | fp32 / eager | 1.0e-07 | exact |
| gpt2-tiny | bf16 / sdpa | 1.2e-05 | exact |
| qwen3-tiny | fp32 / sdpa | 0.0 | exact |
| qwen3-tiny | bf16 / bc250_tiled | 1.5e-05 | exact |
| qwen35-hybrid-tiny | fp32 / sdpa | 1.7e-07 | exact |

### Defects found and fixed while qualifying the fixture

The first host run exposed fixture bugs (not stack incompatibilities); each was
fixed and re-qualified:
1. `state_bytes` used `.numpy()`, which has no bfloat16 support → raw `uint8` view.
2. `chunk_gated_delta_rule` returns a tuple → unpack the output tensor.
3. The CPU/FP64 reference inherited `bc250_tiled` from the deepcopy and failed the
   native dtype check in float64 → reference uses a CPU-compatible `sdpa` path.
4. The frozen-W8 probe converted a trainable `nn.Linear` → freeze the source weight.
5. bf16 compute now uses autocast (the qualified usage pattern for QK-norm models).
6. Battery/probe failures are recorded in the report instead of crashing.

## Verification

- 17 new local tests (`bc250_admission_d62_test.py`); full bc250 suite **190
  passed / 84 subtests**; Black clean (ruff unavailable on the workstation).
- D63: 5/5 admission reports passed; fixture hash bound into each report;
  pre/post container counts 0; no source/provider/prefix/default change, no
  reboot, no foreign process touched. Run via the ordinary `bc250-train` caller
  under the immutable release (no mutable runtime lock involved).

## Scope and next

PLAN-09 is complete for its bounded scope: the matrix documents tested
combinations and the reusable owned admission fixture is qualified. No
model-family promise is made from architecture names alone; each model is admitted
individually with recorded routes/fallbacks. **PLAN-10 (measured training
performance) remains.** Publication of the release still requires explicit user
instruction.