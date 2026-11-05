"""Explicit lora-smoke DDP integration and bounded benchmark instrumentation.

No dataset text or token IDs enter the report. The optional first-update gradient
dump is model-derived numerical evidence, written only to the dedicated output.
"""

import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def verify_saved_adapter(trainer, directory):
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    expected = get_peft_model_state_dict(trainer.model)
    count = 0
    with safe_open(
        str(directory / "adapter_model.safetensors"), framework="pt", device="cpu"
    ) as saved:
        assert set(saved.keys()) == set(expected)
        for key, tensor in expected.items():
            actual = saved.get_tensor(key)
            assert torch.equal(actual, tensor.detach().cpu()), key
            count += tensor.numel()
    assert count == sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    (directory / "bc250-save-verification.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "parameter_tensors": len(expected),
                "elements": count,
                "all_saved_adapter_weights_equal_live_weights": True,
            },
            indent=2,
        )
    )


def configure(args):
    benchmark = os.environ.get("BC250_LORA_BENCHMARK") == "1"
    if benchmark:
        if args.resume or args.allow_existing_output or args.max_steps < 1:
            raise ValueError("Benchmark requires a new output and explicit max_steps")
        if any(args.output_dir.iterdir()) if args.output_dir.exists() else False:
            raise FileExistsError(args.output_dir)
        args.validation_split = 0
        args.eval_data = None
        args.save_steps = args.max_steps + 1
        args.report_to = "none"
        args.epochs = 1
    return benchmark


def configure_trainer(trainer):
    if trainer.accelerator.gradient_accumulation_steps != 1:
        raise RuntimeError("Unexpected Accelerator loss divisor; audit Trainer source")
    # Native public scaler state API, before lazy initialization. This avoids
    # spending a short FP16 benchmark entirely on the default scale's warm-down.
    scale = float(os.environ.get("BC250_LORA_INITIAL_SCALE", "128"))
    if not 0 < scale < float("inf"):
        raise ValueError("Initial loss scale must be finite and positive")
    if trainer.accelerator.scaler is not None:
        state = trainer.accelerator.scaler.state_dict()
        state["scale"] = scale
        trainer.accelerator.scaler.load_state_dict(state)
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, Audit):
            callback.trainer = trainer


class Audit(TrainerCallback):
    def __init__(self, output):
        self.output = Path(output)
        self.rows = []
        self.micro = []
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.trainer = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        trainer = self.trainer
        assert trainer is not None
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        assert self.params and all(
            p.dtype == torch.float32 and p.is_cuda for _, p in self.params
        )
        self.initial = self.digest()
        self.original_training_step = trainer.training_step

        def step(model, inputs, num_items_in_batch=None):
            labels = inputs["labels"]
            hashes = []
            if os.environ.get("BC250_LORA_RECORD_INPUT_HASHES") == "1":
                for tokens, targets, mask in zip(
                    inputs["input_ids"], labels, inputs["attention_mask"]
                ):
                    valid = mask.bool()
                    h = hashlib.sha256()
                    h.update(tokens[valid].detach().cpu().numpy().tobytes())
                    h.update(targets[valid].detach().cpu().numpy().tobytes())
                    hashes.append(h.hexdigest())
            self.micro.append(
                {
                    "example_sha256": hashes,
                    "examples": int(labels.shape[0]),
                    "targets": int((labels[..., 1:] != -100).sum()),
                    "input_tokens": int(inputs["attention_mask"].sum()),
                    "padded_tokens": int(labels.numel()),
                    "denominator": (
                        None if num_items_in_batch is None else int(num_items_in_batch)
                    ),
                }
            )
            return self.original_training_step(model, inputs, num_items_in_batch)

        trainer.training_step = step

    def digest(self):
        digest = hashlib.sha256()
        for name, param in self.params:
            digest.update(name.encode())
            digest.update(param.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        self.started = time.monotonic()
        self.micro = []
        self.scale_before = (
            None
            if self.trainer.accelerator.scaler is None
            else self.trainer.accelerator.scaler.get_scale()
        )

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # This callback runs after native AMP unscale and gradient clipping.
        finite = torch.stack(
            [torch.isfinite(p.grad).all() for _, p in self.params if p.grad is not None]
        ).all()
        self.finite = bool(finite)
        if os.environ.get("BC250_LORA_DUMP_GRADIENT") == "1" and state.global_step == 0:
            schema = []
            path = self.output / f"gradient-rank{self.rank}.f32"
            with path.open("xb") as stream:
                offset = 0
                for name, param in self.params:
                    if param.grad is None:
                        schema.append({"name": name, "absent": True})
                        continue
                    array = param.grad.detach().float().cpu().contiguous().numpy()
                    stream.write(array.tobytes())
                    schema.append(
                        {
                            "name": name,
                            "shape": list(array.shape),
                            "offset": offset,
                            "elements": array.size,
                        }
                    )
                    offset += array.size
            path.with_suffix(".json").write_text(json.dumps(schema))

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.monotonic() - self.started
        skipped = bool(self.trainer.accelerator.optimizer_step_was_skipped)
        stats = torch.tensor(
            [
                sum(m["examples"] for m in self.micro),
                sum(m["targets"] for m in self.micro),
                sum(m["input_tokens"] for m in self.micro),
                sum(m["padded_tokens"] for m in self.micro),
            ],
            device="cuda",
            dtype=torch.int64,
        )
        if dist.is_initialized():
            dist.all_reduce(stats)
        totals = stats.cpu().tolist()
        denominators = [m["denominator"] for m in self.micro]
        assert denominators and all(n == totals[1] for n in denominators), (
            "Global shifted target denominator mismatch",
            denominators,
            totals,
        )
        self.rows.append(
            {
                "step": state.global_step,
                "seconds": elapsed,
                "optimizer_skipped": skipped,
                "finite_gradients": self.finite,
                "scale_before": self.scale_before,
                "examples": totals[0],
                "targets": totals[1],
                "input_tokens": totals[2],
                "padded_tokens": totals[3],
                "microbatches": len(self.micro),
                "example_sha256": [
                    h for micro in self.micro for h in micro["example_sha256"]
                ],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        if os.environ.get("BC250_LORA_BENCHMARK") == "1":
            control.should_save = False

    def on_epoch_end(self, args, state, control, **kwargs):
        if os.environ.get("BC250_LORA_BENCHMARK") == "1":
            control.should_save = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.rows and logs:
            self.rows[-1].setdefault("metrics", {}).update(
                {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            )

    def on_train_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        final = self.digest()
        if dist.is_initialized():
            digests = [None] * self.world
            dist.all_gather_object(digests, final)
            assert len(set(digests)) == 1, "DDP adapter parameters diverged"
        assert final != self.initial, "No adapter update occurred"
        report = {
            "rank": self.rank,
            "world_size": self.world,
            "initial_adapter_sha256": self.initial,
            "final_adapter_sha256": final,
            "trainable_parameters": sum(p.numel() for _, p in self.params),
            "torch": torch.__version__,
            "torch_path": torch.__file__,
            "device": torch.cuda.get_device_properties(0).gcnArchName,
            "accelerator_accumulation": trainer.accelerator.gradient_accumulation_steps,
            "rows": self.rows,
        }
        (self.output / f"audit-rank{self.rank}.json").write_text(
            json.dumps(report, indent=2)
        )
        if os.environ.get("BC250_LORA_BENCHMARK") == "1":
            assert all(
                r["finite_gradients"] and not r["optimizer_skipped"] for r in self.rows
            )
