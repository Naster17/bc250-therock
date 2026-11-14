# PLAN2-06 — broader model admission (owned UNet2D + RNN/LSTM/GRU)

**PASSED 2026-11-14** on .104 (release `10.2-gfx1013-20261111-138a66f52`,
Torch `2.13.0a0+gfx1013`, HIP 7.17.0, boot `0d46a6f1` unchanged). Sixth PLAN2
item. Scope (user-confirmed): admit owned pure-torch **UNet2D (diffusion-style) +
RNN/LSTM/GRU** through the D62-style battery. **No new dependency** (diffusers is
absent from the release), **no app/source/default/package/release change**
(pytorch series stays 23 patches); the D62 fixture itself is untouched (its
qualified helpers are imported).

## Verdict — 6 model/dtype admissions, 60/60 checks

| Model | Params | Dtype | Forward rel-L2 vs CPU-FP64 | Routes fired | Resume |
| --- | ---: | --- | ---: | --- | --- |
| unet2d-tiny | 109171 | FP32 | 0.0 (bitwise) | conv, attention | exact |
| unet2d-tiny | 109171 | BF16 autocast | 3.3e-04 | conv, attention | exact |
| rnn-tiny | 4736 | FP32 | 0.0 (bitwise) | recurrent | exact |
| lstm-tiny | 9536 | FP32 | 0.0 (bitwise) | recurrent | exact |
| lstm-tiny | 9536 | BF16 autocast | 5.0e-04 | recurrent | exact |
| gru-tiny | 7936 | FP32 | 0.0 (bitwise) | recurrent | exact |

Every admission passed the full battery: load → forward vs an independent
CPU-FP64 reference (finite, within D62 tolerance 0.02 FP32 / 0.06 BF16) →
backward (all trainable grads finite, ≥1 nonzero) → AdamW step changes the
bitwise parameter+optimizer fingerprint → **bitwise save/resume** (restored
fingerprint == pre-continuation) → **exact 3-step resumed continuation** (loss
sequence + final fingerprint == live reference). FP32 forwards are bitwise;
BF16 autocast (FP32 masters) is within tolerance. The native recurrent kernels
are deterministic under `cudnn.deterministic=True` (exact resume held).

## What was admitted

- **unet2d-tiny** — a diffusion-style noise-prediction UNet (Conv2d + GroupNorm +
  SiLU + stride-2 down/up-sampling + mid non-causal SDPA self-attention + skip
  connections; MSE noise objective). Exercises the qualified **MIOpen conv** and
  **SDPA attention** routes plus GroupNorm/interpolate. This is a diffusion-style
  UNet trained on noise prediction, **not** a full diffusion pipeline (no
  diffusers, no sampler/scheduler claim).
- **rnn-tiny / lstm-tiny / gru-tiny** — next-token language models (Embedding +
  torch-native `nn.RNN`/`nn.LSTM`/`nn.GRU` + Linear head + cross-entropy).
  Exercise the native **recurrent** kernels.

All models are random-initialized from config; no weights fetched, no
`from_pretrained`/`snapshot_download`, no excluded application. The owned-import
guard passed on every run.

## Method

`build_tools/bc250/admission_plan2_06/` generalizes the D62 battery by
parameterizing the loss/inputs per architecture (the D62 battery is causal-LM
specific). It imports the qualified D62 helpers — `state_bytes` (bitwise
fingerprint), `route_status`, `new_report`/`add_check`/`summarize`,
`record_versions`, `owned_import_guard` — leaving `admit_model_d62.py` unchanged.
Routes are recorded from the modules the architecture actually exercises and
confirmed fired by a successful forward+backward. Each model/dtype runs as its
own bounded guardian child for isolated evidence and failure attribution.

## Attempt journey (harness fix, retained)

ad-r1 crashed: `make_inputs` passed the dtype **string** to `torch.randn`
(fixed: convert via `getattr(torch, dtype)`). ad-r2…ad-r7 then passed all six
admissions. No numeric bound or envelope parameter was relaxed. 11 harness tests,
suite 478, Black clean.

## Envelope and safety

Every run under the qualified PLAN2-01..05 guardian (SH lock, nonce-acked 600 s
boot-guard, 460 s child, `timeout 430` backstop, kernel_noise filter, registry
identity, foreign-container refusal) + stock-clock gate ≤2000 MHz. Boot
`0d46a6f1` unchanged; kernel logs identical (0 benign added); registry unchanged;
0 foreign containers; boards idle pre/post; .105 untouched; host
governor/clock/watchdog/kernel NOT modified. No default/package/release/source
change; not installed into any image.

## Bounds and remaining scope

Admission is **bounded validation, not universal correctness or a model-quality
proof**; forward support alone is not a training claim. Tiny owned geometries
(UNet 32×32×3, recurrent vocab 64/seq 16); no pretrained weights, no real
dataset, no diffusion sampler/scheduler, no large-model throughput claim.
Broader architectures (ViT/ConvNeXt), real diffusion pipelines (diffusers), and
larger geometries remain separate future admissions.

## Reproduction

```bash
python3 build_tools/bc250/admission_plan2_06/run_admission.py --label ad-r2 --model unet2d-tiny --dtype float32
python3 build_tools/bc250/admission_plan2_06/run_admission.py --label ad-r3 --model unet2d-tiny --dtype bfloat16
python3 build_tools/bc250/admission_plan2_06/run_admission.py --label ad-r4 --model rnn-tiny --dtype float32
python3 build_tools/bc250/admission_plan2_06/run_admission.py --label ad-r5 --model lstm-tiny --dtype float32
python3 build_tools/bc250/admission_plan2_06/run_admission.py --label ad-r6 --model lstm-tiny --dtype bfloat16
python3 build_tools/bc250/admission_plan2_06/run_admission.py --label ad-r7 --model gru-tiny --dtype float32
```

## Evidence

`kernel/artifacts/admission-plan2-06/`: BOUNDS.md, admit-ad-r1…r7 reports,
guardians/preflights/kernel logs for every run, orchestrator logs. Host task dir
`/home/nik/bc250work/admission-plan2-06` retained. Publication: bc250-therock
tools+report (no source change), umbrella curated PLAN2-06 + compatibility matrix.