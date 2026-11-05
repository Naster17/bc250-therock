#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual Trainer clipping, coordinated FP16 overflow, final gate and restart."""
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
from transformers import TrainerCallback, TrainingArguments
import transformers
import accelerate
import bc250_distributed_identity
from bc250_trainer_clipping import BC250ClippedTrainer
from bc250_trainer_finite import NonFiniteGradientError
from distributed_d05_train import atomic_json
from distributed_d07_amp import Network, norm_check
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact, fingerprint, rng_state


class TrainerNetwork(Network):
    def forward(self, input_ids, labels=None, row=None):
        return dict(logits=super().forward(input_ids))


class OwnedData(torch.utils.data.Dataset):
    def __init__(self):
        z = torch.arange(28 * 16 * 256).reshape(28, 16, 256)
        self.x = (z % 31 - 15).float() / 32 + torch.arange(28).reshape(-1, 1, 1) / 97
        self.y = (z % 29 - 14).float() / 32 - torch.arange(28).reshape(-1, 1, 1) / 194
        for i in range(28):
            self.y[i, (i % 15) + 1 :, :] = -100

    def __len__(self):
        return 28

    def __getitem__(self, i):
        return dict(input_ids=self.x[i], labels=self.y[i], row=i)


def masked_mse(outputs, labels, num_items_in_batch=None):
    active = labels.ne(-100)
    assert num_items_in_batch is not None and int(num_items_in_batch) > 0
    return (outputs["logits"] - labels).square().masked_select(
        active
    ).sum() / num_items_in_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=["bf16", "fp16", "bf16-reject", "fp16-late", "fp16-scale"],
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
        half = a.case.startswith("fp16")
        reject = a.case not in {"bf16", "fp16"}
        dtype = torch.float16 if half else torch.bfloat16
        torch.manual_seed(557 if a.action == "seed" else 1557)
        base = TrainerNetwork()
        base.accepts_loss_kwargs = False
        oracle = Network()
        oracle.load_state_dict(base.state_dict())
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
            max_grad_norm=0.002,
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
        )
        for layer in base.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.register_forward_hook(
                    lambda module, inputs, output: local["types"].append(
                        str(output.dtype)
                    )
                )

        class ObservedTrainer(BC250ClippedTrainer):
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
                local["rows"].extend(inputs["row"].tolist())
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
                        if a.case in {"fp16", "bf16-reject"}:
                            next(model.parameters()).grad.reshape(-1)[0] = float("inf")
                        elif a.case == "fp16-scale":
                            self.accelerator.scaler.update(
                                new_scale=self.accelerator.scaler.get_scale() / 2
                            )
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
                        if self.state.global_step == 2
                        and a.case in {"fp16", "bf16-reject"}
                        else [True, True]
                    )
                    local["clip_scale"] = scale
                local["losses"].append(float(value))
                assert local["traffic"]["bytes"] - before["bytes"] == (
                    1051648 * 4 if self.accelerator.sync_gradients else 0
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
            compute_loss_func=masked_mse,
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
            parameters=1051648,
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
            max_grad_norm=0.002,
            oracle=digest(Path(__file__).with_name("distributed_d07_amp.py")),
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
                if a.case == "fp16-late" and step == 2 and rank == 1:
                    next(trainer.model.parameters()).grad.reshape(-1)[0] = float("inf")
                flags = [None] * 2
                dist.all_gather_object(
                    flags,
                    all(
                        torch.isfinite(v.grad).all().item()
                        for v in trainer.model.parameters()
                    ),
                )
                assert flags == (
                    [True, False]
                    if a.case == "fp16-late" and step == 2
                    else (
                        [False, False]
                        if a.case == "fp16" and step == 2
                        else [True, True]
                    )
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
                    for actual, wanted in zip(
                        trainer.model.parameters(), oracle.parameters()
                    ):
                        wanted.data.copy_(actual.detach().cpu())
                    oracle.zero_grad(set_to_none=True)
                    cpu_opt.zero_grad(set_to_none=True)
                    denominator = sum(
                        int(data.y[i].ne(-100).sum()) for b in batches for i in b
                    )
                    loss_ref = torch.zeros(())
                    for rows in batches:
                        for start in range(0, len(rows), 2):
                            indices = rows[start : start + 2]
                            output = oracle(data.x[indices], oracle_dtype=dtype)
                            target = data.y[indices]
                            mask = target.ne(-100)
                            loss = (output - target).square().masked_select(
                                mask
                            ).sum() / denominator
                            (loss * 2 * scale).backward()
                            loss_ref += loss.detach()
                    actual_loss = torch.tensor(sum(local["losses"]), device="cuda")
                    dist.all_reduce(actual_loss)
                    actual_loss /= 2
                    torch.testing.assert_close(
                        actual_loss.cpu(),
                        loss_ref,
                        atol=2e-5,
                        rtol=0.002 if not half else 0.0005,
                    )
                    errors = []
                    for actual, wanted, cpu in zip(
                        local["pregrads"],
                        oracle.parameters(),
                        cpu_model.parameters(),
                    ):
                        errors.append(
                            norm_check(
                                actual,
                                wanted.grad / 2 / scale,
                                0.004 if half else 0.02,
                            )
                        )
                        cpu.grad = actual.double().clone()
                    # Independent FP64 global L2 norm and clipping coefficient;
                    # CPU AdamW consumes these expected clipped gradients.
                    norm = math.sqrt(
                        sum(v.double().square().sum().item() for v in local["pregrads"])
                    )
                    coefficient = min(1.0, 0.002 / (norm + 1e-6))
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
                    r["gradient_errors"].append(
                        dict(step=step, maximum_relative_L2=max(errors))
                    )
                    r["checks"].append(f"CPU-rounding-global-loss-all-gradients-{step}")

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
                        65536 if ts.global_step <= 2 else 32768
                    )
                    assert acc.scaler.state_dict()["_growth_tracker"] == (
                        ts.global_step if ts.global_step <= 2 else ts.global_step - 3
                    )
                value = dict(
                    state=state(),
                    grads=local["grads"],
                    rows=local["rows"],
                    losses=local["losses"],
                    draws=local["draws"],
                    finite=local["finite"],
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
                    )
                )
                local["rows"] = []
                local["losses"] = []
                local["draws"] = []
                local["types"] = []

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
        try:
            outcome = trainer.train(
                resume_from_checkpoint=str(checkpoint) if a.action == "resume" else None
            )
        except NonFiniteGradientError:
            assert reject and trainer.state.global_step == 2 and successful() == 2
            expected = torch.load(
                a.seed_directory / "checkpoint-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            for key in ["params", "optimizer", "scheduler"]:
                exact(state()[key], expected[key])
            refused = [None] * 2
            dist.all_gather_object(refused, True)
            assert refused == [True, True]
            r["checks"].append(
                "rank1-local-"
                + a.case
                + "-collective-refusal-before-parameter-optimizer-scheduler-mutation"
            )
        else:
            assert not reject and outcome.global_step == 4
            assert successful() == (3 if half else 4)
            assert raw_calls[0] == (
                (3 if half else 4) if a.action == "seed" else (1 if half else 2)
            )
            r["checks"].append(
                "actual-Trainer-AMP-DDP-short-final-window-and-update-only-StepLR-completed"
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
