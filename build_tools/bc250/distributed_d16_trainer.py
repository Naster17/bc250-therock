#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned causal GPT2 Trainer AMP, shifted CE/counts, clipping and native restart."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist
from transformers import GPT2Config, GPT2LMHeadModel, TrainerCallback, TrainingArguments
from accelerate.utils import GradScalerKwargs
from bc250_trainer_causal import CausalLabelCollator, causal_token_loss
from check_causal_d16 import contract_checks
from transformers.pytorch_utils import Conv1D
import transformers
import accelerate
import bc250_distributed_identity
from bc250_trainer_clipping import BC250ClippedTrainer
from distributed_d05_train import atomic_json
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact, fingerprint, rng_state


class OwnedData(torch.utils.data.Dataset):
    def __init__(self):
        self.ids = (torch.arange(28 * 12).reshape(28, 12) * 7 + 3) % 96 + 1
        self.ids[:, 0] = torch.arange(28) + 1
        self.mask = torch.ones_like(self.ids)
        self.labels = self.ids.clone()
        for i in range(28):
            length = 5 + i % 8
            self.mask[i, length:] = 0
            self.ids[i, length:] = 0
            self.labels[i, length:] = -100
            self.labels[i, : i % 3] = -100
        self.labels[:2, :] = (
            -100
        )  # One real rank-local micro has no supervised targets.
        self.targets = self.labels.clone()
        self.targets[:, 0] = -100
        self.targets.masked_fill_(self.mask == 0, -100)

    def __len__(self):
        return 28

    def __getitem__(self, i):
        return dict(
            input_ids=self.ids[i], attention_mask=self.mask[i], labels=self.labels[i]
        )


def cpu_token_ce(logits, labels, denominator):
    active = labels[:, 1:].ne(-100)
    selected = logits[:, :-1, :][active]
    targets = labels[:, 1:][active]
    return (
        selected.logsumexp(-1) - selected.gather(1, targets[:, None]).squeeze(1)
    ).sum() / denominator


def global_relative(actual, expected):
    error = math.sqrt(
        sum(
            (x.double() - y.double()).square().sum().item()
            for x, y in zip(actual, expected)
        )
    )
    norm = math.sqrt(sum(y.double().square().sum().item() for y in expected))
    return error / max(norm, 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=["bf16", "fp16"],
        required=True,
    )
    p.add_argument("--action", choices=["seed", "resume"], required=True)
    p.add_argument("--seed-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(
        status="running",
        case=a.case,
        action=a.action,
        checks=[],
        transcript=[],
        gradient_errors=[],
    )
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        r["CPU_contract_checks"] = contract_checks()
        half = a.case.startswith("fp16")
        dtype = torch.float16 if half else torch.bfloat16
        torch.manual_seed(773 if a.action == "seed" else 1773)
        model_config = GPT2Config(
            vocab_size=97,
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_inner=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
            attn_implementation="eager",
        )
        base = GPT2LMHeadModel(model_config)
        base.accepts_loss_kwargs = False
        oracle = copy.deepcopy(base).float()
        oracle64 = copy.deepcopy(base).double()
        cpu_model = copy.deepcopy(base).double()
        opt_raw = torch.optim.AdamW(
            base.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )
        cpu_opt = torch.optim.AdamW(
            cpu_model.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )
        scheduler = torch.optim.lr_scheduler.StepLR(opt_raw, step_size=2, gamma=0.8)
        data = OwnedData()
        args = TrainingArguments(
            output_dir=str(a.output.parent / "trainer-output"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=4,
            max_grad_norm=0.05,
            report_to="none",
            bf16=not half,
            fp16=half,
            save_strategy="steps",
            save_steps=2,
            save_on_each_node=True,
            save_total_limit=None,
            logging_strategy="no",
            eval_strategy="no",
            disable_tqdm=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            train_sampling_strategy="sequential",
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
            seed=911,
            data_seed=1011,
            label_names=["labels"],
            average_tokens_across_devices=True,
            full_determinism=True,
            accelerator_config={
                "split_batches": False,
                "gradient_accumulation_kwargs": {"adjust_scheduler": False},
            },
        )
        birth = dict(
            uuid=uuid.uuid4().hex,
            pid=os.getpid(),
            start_ticks=Path("/proc/self/stat")
            .read_text()
            .split(") ", 1)[1]
            .split()[19],
        )
        raw_calls = [0]
        opt_raw.register_step_post_hook(
            lambda *_: raw_calls.__setitem__(0, raw_calls[0] + 1)
        )
        local = dict(
            rows=[],
            losses=[],
            draws=[],
            types=[],
            traffic=dict(calls=0, bytes=0),
            registered=False,
            counts=[],
            local_active=[],
        )
        for layer in base.modules():
            if isinstance(layer, (torch.nn.Linear, Conv1D)):
                layer.register_forward_hook(
                    lambda module, inputs, output: local["types"].append(
                        str(output.dtype)
                    )
                )

        def observed_token_loss(outputs, labels, num_items_in_batch=None):
            assert outputs.logits.dtype == torch.float32
            local["counts"].append(int(num_items_in_batch))
            active = int(labels[:, 1:].ne(-100).sum())
            local["local_active"].append(active)
            value = causal_token_loss(outputs, labels, num_items_in_batch)
            assert torch.isfinite(value).item()
            if active == 0:
                assert float(value) == 0
                r["checks"].append(
                    "GPU-local-zero-target-micro-positive-global-finite-zero-loss"
                )
            return value

        class ObservedTrainer(BC250ClippedTrainer):
            def _build_accelerator_args(self, **kwargs):
                result = super()._build_accelerator_args(**kwargs)
                result["kwargs_handlers"] = [
                    GradScalerKwargs(init_scale=128, growth_interval=2)
                ]
                return result

            def training_step(self, model, inputs, num_items_in_batch=None):
                if not local["registered"]:
                    assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
                    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
                        allreduce_hook,
                    )

                    def hook(state, bucket):
                        state["calls"] += 1
                        state["bytes"] += (
                            bucket.buffer().numel() * bucket.buffer().element_size()
                        )
                        assert bucket.buffer().dtype == torch.float32
                        return allreduce_hook(None, bucket)

                    model.register_comm_hook(local["traffic"], hook)
                    local["registered"] = True
                local["rows"].extend((inputs["input_ids"][:, 0] - 1).tolist())
                local["draws"].append(
                    [
                        random.random(),
                        float(np.random.random()),
                        torch.rand(()).item(),
                        torch.rand((), device="cuda").item(),
                    ]
                )
                before = dict(local["traffic"])
                value = super().training_step(model, inputs, num_items_in_batch)
                if self.accelerator.sync_gradients:
                    if self.state.global_step == 2 and dist.get_rank() == 1:
                        if a.case == "fp16":
                            next(model.parameters()).grad.reshape(-1)[0] = float("inf")
                    scale = (
                        1
                        if self.accelerator.scaler is None
                        else self.accelerator.scaler.get_scale()
                    )
                    local["pregrads"] = [
                        cpu_tree(v.grad / scale) for v in model.parameters()
                    ]
                    flags = [None] * 2
                    dist.all_gather_object(
                        flags,
                        all(
                            torch.isfinite(v.grad).all().item()
                            for v in model.parameters()
                        ),
                    )
                    assert flags == (
                        [True, False]
                        if self.state.global_step == 2 and a.case == "fp16"
                        else [True, True]
                    )
                    local["clip_scale"] = scale
                local["losses"].append(float(value))
                assert local["traffic"]["bytes"] - before["bytes"] == (
                    21280 * 4 if self.accelerator.sync_gradients else 0
                )
                assert set(local["types"]) == {str(dtype)}
                return value

            def create_accelerator_and_postprocess(self):
                super().create_accelerator_and_postprocess()
                native = self.accelerator.clip_grad_norm_

                def observed(parameters, max_norm, norm_type=2):
                    value = native(parameters, max_norm, norm_type)
                    local["norm"] = float(value)
                    return value

                self.accelerator.clip_grad_norm_ = observed

            def _load_rng_state(self, checkpoint):
                super()._load_rng_state(checkpoint)
                if checkpoint:
                    exact(
                        state(),
                        torch.load(
                            a.seed_directory / "checkpoint-state.pt",
                            map_location="cpu",
                            weights_only=True,
                        ),
                    )
                    r["checks"].append(
                        "native-Trainer-initial-load-params-AdamW-scheduler-scaler-RNG-step-exact"
                    )

            def _save_checkpoint(self, model, trial):
                super()._save_checkpoint(model, trial)
                if self.state.global_step == 2:
                    torch.save(state(), a.seed_directory / "checkpoint-state.pt")
                    ckpt = Path(self.args.output_dir) / "checkpoint-2"
                    files = [
                        dict(name=v.name, bytes=v.stat().st_size, sha256=digest(v))
                        for v in sorted(ckpt.iterdir())
                        if v.is_file()
                    ]
                    names = {v["name"] for v in files}
                    assert {
                        "model.safetensors",
                        "optimizer.pt",
                        "scheduler.pt",
                        "trainer_state.json",
                        f"rng_state_{self.args.process_index}.pth",
                    }.issubset(names)
                    if half:
                        assert "scaler.pt" in names
                    atomic_json(
                        a.seed_directory / "complete.json",
                        dict(
                            config=config,
                            rank=self.args.process_index,
                            birth=birth,
                            files=files,
                        ),
                    )
                    r["checks"].append(
                        "native-Trainer-checkpoint2-complete-on-each-host"
                    )

        trainer = ObservedTrainer(
            model=base,
            args=args,
            train_dataset=data,
            optimizers=(opt_raw, scheduler),
            compute_loss_func=observed_token_loss,
            data_collator=CausalLabelCollator(),
        )
        acc = trainer.accelerator
        rank = acc.process_index
        assert (
            acc.num_processes == 2
            and acc.distributed_type.value == "MULTI_GPU"
            and acc.state.backend == "nccl"
        )
        assert acc.gradient_accumulation_steps == 1 and (acc.scaler is not None) == half
        assert acc.mixed_precision == a.case.split("-")[0]
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        tool = Path(bc250_distributed_identity.__file__).parent
        package = Path(transformers.__file__).parent
        config = dict(
            case=a.case,
            world=2,
            parameters=21280,
            global_batch=8,
            accumulation=2,
            records=28,
            workers=0,
            transformers=transformers.__version__,
            accelerate=accelerate.__version__,
            fixture=digest(Path(__file__)),
            compat=digest(Path(__file__).with_name("bc250_trainer.py")),
            guard=digest(Path(__file__).with_name("bc250_trainer_finite.py")),
            clipping=digest(Path(__file__).with_name("bc250_trainer_clipping.py")),
            max_grad_norm=0.05,
            causal_helper=digest(Path(__file__).with_name("bc250_trainer_causal.py")),
            contract_helper=digest(Path(__file__).with_name("check_causal_d16.py")),
            gpt2=digest(package / "models/gpt2/modeling_gpt2.py"),
            reference="CPU native autocast micro/rank +manual logsumexp CE and FP64 global model",
            reference_limits=dict(
                AMP_relativeL2=0.004 if half else 0.02,
                FP64_relativeL2=0.006 if half else 0.03,
                per_parameter_floor=3e-7 if half else 3e-6,
            ),
            fp16_init_scale=128,
            fp16_growth_interval=2,
            native=digest(tool / "torch-identity.json"),
            trainer=digest(package / "trainer.py"),
        )
        r.update(rank=rank, birth=birth, config=config, source_sha256=config["fixture"])

        def state():
            return dict(
                params=[cpu_tree(v) for v in trainer.model.parameters()],
                optimizer=cpu_tree(trainer.optimizer.state_dict()),
                scheduler=cpu_tree(trainer.lr_scheduler.state_dict()),
                scaler={} if acc.scaler is None else acc.scaler.state_dict(),
                rng=rng_state(),
                global_step=trainer.state.global_step,
            )

        def successful():
            return (
                int(next(iter(opt_raw.state.values()))["step"]) if opt_raw.state else 0
            )

        class Checks(TrainerCallback):
            def on_pre_optimizer_step(self, args, ts, control, **kwargs):
                step = ts.global_step
                batches = [None] * 2
                dist.all_gather_object(batches, list(local["rows"]))
                assert sorted(i for b in batches for i in b) == list(
                    range(step * 8, min(step * 8 + 8, 28))
                )
                expected_count = sum(
                    int(data.targets[i, 1:].ne(-100).sum())
                    for part in batches
                    for i in part
                )
                raw_count = sum(
                    int(data.labels[i].ne(-100).sum()) for part in batches for i in part
                )
                assert local["counts"] and set(local["counts"]) == {expected_count}
                r.setdefault("target_counts", []).append(
                    dict(
                        step=step,
                        shifted=expected_count,
                        raw_unshifted=raw_count,
                        local_active=list(local["local_active"]),
                    )
                )
                r["checks"].append(f"native-Trainer-global-shifted-target-count-{step}")
                flags = [None] * 2
                dist.all_gather_object(
                    flags,
                    all(
                        torch.isfinite(v.grad).all().item()
                        for v in trainer.model.parameters()
                    ),
                )
                assert flags == (
                    [False, False] if a.case == "fp16" and step == 2 else [True, True]
                ), flags
                local["finite"] = all(flags)
                scale = local["clip_scale"]
                local["grads"] = [
                    dict(
                        values=cpu_tree(
                            torch.nan_to_num(v.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        ),
                        nan=cpu_tree(torch.isnan(v.grad)),
                        posinf=cpu_tree(torch.isposinf(v.grad)),
                        neginf=cpu_tree(torch.isneginf(v.grad)),
                    )
                    for v in trainer.model.parameters()
                ]
                local["before"] = dict(
                    params=[cpu_tree(v) for v in trainer.model.parameters()],
                    optimizer=cpu_tree(opt_raw.state_dict()),
                    scheduler=cpu_tree(trainer.lr_scheduler.state_dict()),
                )
                local["calls"] = raw_calls[0]
                if a.action == "seed" and local["finite"]:
                    for actual, wanted, wanted64 in zip(
                        trainer.model.parameters(),
                        oracle.parameters(),
                        oracle64.parameters(),
                    ):
                        wanted.data.copy_(actual.detach().cpu())
                        wanted64.data.copy_(actual.detach().cpu().double())
                    oracle.zero_grad(set_to_none=True)
                    oracle64.zero_grad(set_to_none=True)
                    cpu_opt.zero_grad(set_to_none=True)
                    denominator = sum(
                        int(data.targets[i, 1:].ne(-100).sum())
                        for b in batches
                        for i in b
                    )
                    loss_ref = torch.zeros(())
                    for rows in batches:
                        for start in range(0, len(rows), 2):
                            indices = rows[start : start + 2]
                            with torch.autocast("cpu", dtype=dtype):
                                logits = oracle(
                                    input_ids=data.ids[indices],
                                    attention_mask=data.mask[indices],
                                ).logits
                            assert logits.dtype == dtype
                            loss = cpu_token_ce(
                                logits.float(), data.targets[indices], denominator
                            )
                            (loss * 2 * scale).backward()
                            loss_ref += loss.detach()
                    rows = [i for part in batches for i in part]
                    logits64 = oracle64(
                        input_ids=data.ids[rows], attention_mask=data.mask[rows]
                    ).logits
                    loss64 = cpu_token_ce(logits64, data.targets[rows], denominator)
                    loss64.backward()
                    actual_loss = torch.tensor(sum(local["losses"]), device="cuda")
                    dist.all_reduce(actual_loss)
                    actual_loss /= 2
                    torch.testing.assert_close(
                        actual_loss.cpu(),
                        loss_ref,
                        atol=3e-3 if not half else 1e-3,
                        rtol=3e-3 if not half else 1e-3,
                    )
                    torch.testing.assert_close(
                        actual_loss.cpu().double(),
                        loss64.detach(),
                        atol=3e-3 if not half else 1e-3,
                        rtol=3e-3 if not half else 1e-3,
                    )
                    amp_grads = [v.grad / 2 / scale for v in oracle.parameters()]
                    double_grads = [v.grad for v in oracle64.parameters()]
                    relative_amp = global_relative(local["pregrads"], amp_grads)
                    relative64 = global_relative(local["pregrads"], double_grads)
                    assert relative_amp <= (0.004 if half else 0.02), relative_amp
                    assert relative64 <= (0.006 if half else 0.03), relative64
                    # Absolute norm floors account for near-zero attention key-bias
                    # cancellation. Relative bounds also apply to every parameter.
                    floor = 3e-7 if half else 3e-6
                    for actual, amp, wanted64, cpu in zip(
                        local["pregrads"],
                        amp_grads,
                        double_grads,
                        cpu_model.parameters(),
                    ):
                        for expected, limit in [
                            (amp, 0.004 if half else 0.02),
                            (wanted64, 0.006 if half else 0.03),
                        ]:
                            error = (actual.double() - expected.double()).norm().item()
                            bound = (
                                limit * expected.double().norm().item()
                                + floor * math.sqrt(expected.numel())
                            )
                            assert error <= bound, (
                                step,
                                error,
                                bound,
                                tuple(actual.shape),
                            )
                        cpu.grad = actual.double().clone()
                    r["gradient_errors"].append(
                        dict(
                            step=step,
                            global_AMP_relativeL2=relative_amp,
                            global_FP64_relativeL2=relative64,
                        )
                    )
                    r["checks"].append(
                        f"CPU-AMP-and-FP64-global-SHIFTED-CE-all-gradients-{step}"
                    )
                    # Independent FP64 global L2 norm and clipping coefficient;
                    # CPU AdamW consumes these expected clipped gradients.
                    norm = math.sqrt(
                        sum(v.double().square().sum().item() for v in local["pregrads"])
                    )
                    coefficient = min(1.0, 0.05 / (norm + 1e-6))
                    assert coefficient < 0.95, (norm, coefficient)
                    assert math.isclose(local["norm"], norm, rel_tol=1e-5, abs_tol=1e-7)
                    for actual, cpu in zip(
                        trainer.model.parameters(), cpu_model.parameters()
                    ):
                        cpu.grad.mul_(coefficient)
                        torch.testing.assert_close(
                            actual.grad.detach().cpu().double(),
                            cpu.grad,
                            atol=1e-8,
                            rtol=5e-5,
                        )
                    r.setdefault("clipping_metrics", []).append(
                        dict(step=step, norm=norm, coefficient=coefficient)
                    )
                    r["checks"].append(f"CPU-FP64-global-L2-clip-all-gradients-{step}")

            def on_optimizer_step(self, args, ts, control, **kwargs):
                assert raw_calls[0] - local["calls"] == int(local["finite"])
                assert (
                    acc.optimizer_step_was_skipped == (not local["finite"])
                    if half
                    else not acc.optimizer_step_was_skipped
                )
                if not local["finite"]:
                    exact(
                        dict(
                            params=[cpu_tree(v) for v in trainer.model.parameters()],
                            optimizer=cpu_tree(opt_raw.state_dict()),
                            scheduler=cpu_tree(trainer.lr_scheduler.state_dict()),
                        ),
                        local["before"],
                    )
                    r["checks"].append(
                        "native-FP16-skips-AdamW-and-preserves-parameters-moments-scheduler"
                    )
                elif a.action == "seed":
                    cpu_opt.step()
                    for actual, wanted in zip(
                        trainer.model.parameters(), cpu_model.parameters()
                    ):
                        torch.testing.assert_close(
                            actual.detach().cpu().double(), wanted, atol=4e-6, rtol=5e-5
                        )
                        for key, atol, rtol in [
                            ("exp_avg", 3e-7, 5e-4),
                            ("exp_avg_sq", 3e-9, 1e-3),
                        ]:
                            torch.testing.assert_close(
                                opt_raw.state[actual][key].cpu().double(),
                                cpu_opt.state[wanted][key],
                                atol=atol,
                                rtol=rtol,
                            )
                    r["checks"].append(
                        f"CPU-AdamW-all-parameters-moments-{ts.global_step}"
                    )

            def on_step_end(self, args, ts, control, **kwargs):
                updates = successful()
                assert trainer.lr_scheduler.state_dict()["last_epoch"] == updates
                assert math.isclose(
                    trainer.optimizer.param_groups[0]["lr"],
                    0.001 * 0.8 ** (updates // 2),
                    rel_tol=0,
                    abs_tol=1e-15,
                )
                for group in cpu_opt.param_groups:
                    group["lr"] = 0.001 * 0.8 ** (updates // 2)
                scalers = [None] * 2
                dist.all_gather_object(
                    scalers, {} if acc.scaler is None else acc.scaler.state_dict()
                )
                assert scalers[0] == scalers[1]
                if half:
                    assert acc.scaler.get_scale() == (
                        (128 if ts.global_step == 1 else 256)
                        if ts.global_step <= 2
                        else 128
                    )
                    assert acc.scaler.state_dict()["_growth_tracker"] == (
                        (1 if ts.global_step == 1 else 0)
                        if ts.global_step <= 2
                        else ts.global_step - 3
                    )
                value = dict(
                    state=state(),
                    grads=local["grads"],
                    rows=local["rows"],
                    losses=local["losses"],
                    draws=local["draws"],
                    finite=local["finite"],
                    target_counts=local["counts"],
                    local_active=local["local_active"],
                )
                if ts.global_step > 2:
                    path = a.seed_directory / f"reference-{ts.global_step}.pt"
                    if a.action == "seed":
                        torch.save(value, path)
                    else:
                        exact(
                            value,
                            torch.load(path, map_location="cpu", weights_only=True),
                        )
                        r["checks"].append(
                            f"fresh-native-Trainer-replay-exact-all-state-{ts.global_step}"
                        )
                r["transcript"].append(
                    dict(
                        step=ts.global_step,
                        digest=fingerprint(value),
                        updates=updates,
                        finite=local["finite"],
                        target_counts=local["counts"],
                        local_active=local["local_active"],
                    )
                )
                local["rows"] = []
                local["losses"] = []
                local["draws"] = []
                local["types"] = []
                local["counts"] = []
                local["local_active"] = []

        trainer.add_callback(Checks())
        checkpoint = a.seed_directory / "trainer-output/checkpoint-2"
        if a.action == "resume":
            header = json.loads((a.seed_directory / "complete.json").read_text())
            assert header["rank"] == rank and header["config"] == config
            for f in header["files"]:
                path = checkpoint / f["name"]
                assert path.stat().st_size == f["bytes"] and digest(path) == f["sha256"]
            assert (
                header["birth"]["uuid"] != birth["uuid"]
                and header["birth"]["start_ticks"] != birth["start_ticks"]
            )
            r["source_birth"] = header["birth"]
        outcome = trainer.train(
            resume_from_checkpoint=str(checkpoint) if a.action == "resume" else None
        )
        assert outcome.global_step == 4 and successful() == (3 if half else 4)
        assert raw_calls[0] == (
            (3 if half else 4) if a.action == "seed" else (1 if half else 2)
        )
        r["checks"].append(
            "actual-causal-Transformer-Trainer-AMP-shifted-CE-active-clipping-short-window-and-StepLR-completed"
        )
        r.update(
            final_digest=fingerprint(state()),
            optimizer_updates=successful(),
            global_step=trainer.state.global_step,
            final_scaler={} if acc.scaler is None else acc.scaler.state_dict(),
            status="passed",
        )
        dist.barrier()
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        atomic_json(a.output, r)


if __name__ == "__main__":
    main()
