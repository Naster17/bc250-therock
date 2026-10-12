# BC-250 short training and memory qualification

Qualified on `.104` on 2026-10-13 using the installed expanded release
`10.2-gfx1013-20261013-4b000d542`, native `gfx1013`, and SDMA disabled.
Performance benchmarking and tuning were not started. `.105` was unchanged.

The successful final batch completed nine gates in **123.99 seconds**. Package
setup and the first download are separate from that duration. No new kernel
messages appeared during the batch. This is bounded functional qualification,
not a long-duration stability or model-quality claim.

## Pretrained model and isolated packages

The gate uses [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
with **134,515,008 base parameters**, pinned revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The weights match SHA-256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
Tokenizer/configuration hashes are recorded in `training-model/bc250-model-provenance.json`.

The persistent consumer environment contains:

| Package          | Version                                           |
| ---------------- | ------------------------------------------------- |
| Torch            | 2.13.0a0+gfx1013, supplied by the validated image |
| Triton           | 3.8.0, supplied by the validated image            |
| Transformers     | 5.2.0                                             |
| PEFT             | 0.18.1                                            |
| Accelerate       | 1.12.0                                            |
| Hugging Face Hub | 1.4.1                                             |

`setup_training_env.py` creates a host-mounted glibc Python venv that can see the
validated runtime's packages. Constraints retain Torch, Triton, NumPy and Click;
`pip check` passed. Consumer packages are selected through a `PYTHONPATH` env file.
The image, installed ROCm libraries and fallback release were not modified.

The first attempted newer package set required a Click version incompatible with
the image's build tooling. A compatible pinned set was selected instead; the
failed attempt is not the qualified environment.

Use the prepared environment from an SSH shell on `.104`:

```bash
bc250-torch --env-file \
  ~/bc250work/rocm-10.2-gfx1013/training-env-stable/training.env \
  your_training_script.py
```

The glibc venv belongs inside the container; its interpreter is not an Alpine
host Python environment. No Transformers/PEFT installation replaces the custom
Torch build. The qualified environment's principal resolved package versions are in
`training-env-stable/packages.json`.

## Full fine-tuning and AMP LoRA

All modes used 32 tokens, one local text example, mathematical SDPA and AdamW.
Each performed four updates, saved a checkpoint, then resumed for one update in
an independent container. These checks establish execution and short loss
improvement on that example, not downstream training quality.

| Mode                    | Trainable parameters | FP32 evaluation loss: initial → four updates → resumed update | Peak Torch allocation during training |
| ----------------------- | -------------------: | ------------------------------------------------------------- | ------------------------------------: |
| Full FP32 fine-tuning   |          134,515,008 | 4.647392 → 2.845694 → 2.507597                                |                   2,921,034,752 bytes |
| LoRA with FP16 autocast |              460,800 | 4.647392 → 4.017544 → 3.802662                                |                     704,305,152 bytes |
| LoRA with BF16 autocast |              460,800 | 4.647392 → 4.017366 → 3.803379                                |                     704,303,616 bytes |

LoRA used rank 8, alpha 16, zero adapter dropout, and `q_proj`/`v_proj` targets.
Base parameters remained frozen; representative base projection weights were
verified unchanged. Both LoRA modes exercised non-reentrant activation
checkpointing. FP16 used GradScaler with initial scale 128; gradients stayed
finite and no optimizer update was skipped because of AMP overflow.

The model/adapter and optimizer, scheduler, scaler, CPU RNG and GPU RNG states
were saved. Fresh-container resume checked saved logits, exact RNG probes and
continued optimizer step counts. Adapter configuration records the public base
model ID and its pinned revision. PEFT's separate adapter/base format is described
in the [official checkpoint documentation](https://huggingface.co/docs/peft/en/developer_guides/checkpoint).

The full FP32 gate additionally compared CPU/GPU logits, loss and gradients for
embedding, attention projection, MLP projection and final normalization weights:

- Logit relative L2 error: **5.5637e-6**.
- Maximum absolute logit difference: **0.00045347**.
- Gradient relative L2 errors: **8.36e-5, 2.58e-4, 3.21e-4, 4.68e-4**.

Initial tight elementwise thresholds rejected a handful of near-zero logits and
gradients. The final gate records aggregate and maximum errors, bounds logit
relative L2 below 2e-5, and gradient relative L2 below 2e-3, with independent
maximum-error and loss checks. It accounts for deep FP32 accumulation rather
than requiring near-zero components to match bit-for-bit. The original failed
comparison logs are retained; no ROCm runtime change was made to hide them.

## Shared memory, streams, graphs and processes

The memory gate accounts for shared CPU/GPU physical RAM before allocating. It
caps the requested GPU pressure at 10 GiB and budgets 3 GiB for system/scratch
headroom rather than deliberately exhausting the SSH-only host.

- **10 GiB of GPU tensors plus 768 MiB of CPU tensors** were filled and every
  byte verified.
- Alternating GPU allocations were freed and reused while remaining live
  buffers retained their expected values.
- GPU peak allocation was **11,005,856,768 bytes**, including temporary checks.
- After release and `empty_cache`, the pressure case reported **zero allocated
  and zero reserved bytes**.
- Four streams exercised pinned asynchronous copies, events and CPU-compared
  results over 12 iterations.
- HIP graph capture passed eight replays with changing input and CPU references.
- Two independent GPU processes ran together, each checking GEMM against FP64
  CPU accumulation and preserving a distinct 256 MiB live buffer through 12
  allocation/free iterations.

Each concurrent process retained the expected **32 MiB BLAS workspace** while
alive. The pinned Torch source's `CublasHandlePool.cpp` defines that default for
this HIP architecture. Both processes exited cleanly.

These gates used the compiled gfx1013 safe allocator default with no VM-heap
workaround environment override. The experimental VM heap remains disabled by
default; repairing that separate path is future work.

## Artifacts and reproduction

On `.104`, relative to `/home/nik/bc250work/rocm-10.2-gfx1013`:

- `training-env-stable/training.env`: launcher environment for the installed consumers.
- `training-model/`: pinned pretrained weights/tokenizer and checksums.
- `training-validation/full/model/`: fully fine-tuned model snapshot.
- `training-validation/lora-fp16/adapter/` and `lora-bf16/adapter/`: portable adapters.
- `training-validation/<mode>/training-state.pt`: optimizer/scheduler/scaler/RNG state.
- `training-validation/<mode>/train.json` and `resume.json`: losses and validation details.
- `logs/training-memory/result.json`: nine-gate result and explicit Performance boundary.
- `logs/training-memory/`: pressure/stream/process reports and before/after kernel logs.

To resume the prepared FP16 LoRA example for one validated update:

```bash
bc250-torch --env-file \
  ~/bc250work/rocm-10.2-gfx1013/training-env-stable/training.env \
  --env HF_HUB_OFFLINE=1 \
  ~/bc250work/rocm-10.2-gfx1013/helpers/validate_model_training.py \
  --model ~/bc250work/rocm-10.2-gfx1013/training-model \
  --directory ~/bc250work/rocm-10.2-gfx1013/training-validation \
  --mode lora-fp16 --phase resume
```

The host orchestrator is `run_training_memory_on_host.py`. Its `--resume` option
reuses previously passed cases after an unrelated failed case is fixed; do not
use it to bypass requalification of behavior changed by a patch. Per-case
execution is bounded, and timeout handling first forwards graceful termination.

## Remaining work at the stop boundary

Performance measurements, `.105` comparisons and tuning are the next stage.
They were deliberately left untouched. The following deeper qualifications also
remain explicitly pending:

- Larger models, longer sequences/batches, real datasets and multi-hour convergence.
- Full-model AMP variants, further LoRA targets and adapter types, QLoRA/4-bit backends.
- Full-model eager/compiled agreement and training graph capture with dynamic shapes.
- Near-capacity/OOM recovery, graph allocation/free nodes, IPC and explicit pool/VMM APIs.
- GPU fault/reset recovery and watchdog behavior under deliberate faults.
- Distributed training/RCCL, broader precision/operator coverage and all remaining
  ROCm ports in [bc250_full_support_roadmap.md](bc250_full_support_roadmap.md).
