#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual Accelerate DDP/data/AMP/scheduler and fresh native save_state/load_state replay."""
import argparse
import copy
from datetime import timedelta
import itertools
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import (
    DistributedDataParallelKwargs,
    GradScalerKwargs,
    GradientAccumulationPlugin,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)
import accelerate
import bc250_distributed_identity
from distributed_d05_train import atomic_json
from distributed_d07_amp import Network, norm_check
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact, fingerprint, rng_state


class Progress:
    def __init__(self):
        self.micros = 0
        self.updates = 0

    def state_dict(self) -> dict:
        return dict(micros=self.micros, updates=self.updates)

    def load_state_dict(self, value: dict) -> None:
        self.micros = value["micros"]
        self.updates = value["updates"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", choices=["no", "bf16", "fp16"], required=True)
    p.add_argument("--action", choices=["seed", "resume"], required=True)
    p.add_argument("--seed-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = dict(
        status="running",
        precision=a.precision,
        action=a.action,
        checks=[],
        transcript=[],
        gradient_errors=[],
    )
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        acc = Accelerator(
            mixed_precision=a.precision,
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=4, adjust_scheduler=False
            ),
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            project_config=ProjectConfiguration(
                save_on_each_node=True, automatic_checkpoint_naming=False
            ),
            kwargs_handlers=[
                DistributedDataParallelKwargs(gradient_as_bucket_view=True),
                GradScalerKwargs(init_scale=16, growth_interval=2),
                InitProcessGroupKwargs(timeout=timedelta(seconds=120)),
            ],
        )
        rank, world = acc.process_index, acc.num_processes
        assert (
            world == 2
            and acc.local_process_index == 0
            and acc.distributed_type.value == "MULTI_GPU"
        )
        assert (
            acc.device.type == "cuda"
            and acc.state.backend == "nccl"
            and acc.mixed_precision == a.precision
        )
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert len(os.sched_getaffinity(0)) <= 4
        birth = dict(
            uuid=uuid.uuid4().hex,
            pid=os.getpid(),
            start_ticks=Path("/proc/self/stat")
            .read_text()
            .split(") ", 1)[1]
            .split()[19],
        )
        dtype = {"no": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            a.precision
        ]
        torch.manual_seed(557 if a.action == "seed" else 1557 + rank)
        base = Network()
        oracle = (
            copy.deepcopy(base).double() if a.precision == "no" else copy.deepcopy(base)
        )
        cpu_opt_model = copy.deepcopy(base).double()
        cpu_opt = torch.optim.AdamW(
            cpu_opt_model.parameters(),
            lr=0.001,
            eps=1e-5,
            weight_decay=0.01,
            foreach=False,
        )
        progress = Progress()
        opt_raw = torch.optim.AdamW(
            base.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )

        def updated(*_) -> None:
            progress.updates += 1

        opt_raw.register_step_post_hook(updated)
        sched_raw = torch.optim.lr_scheduler.StepLR(opt_raw, step_size=2, gamma=0.8)
        shape = (32, 16, 256)
        indices = torch.arange(math.prod(shape)).reshape(shape)
        data = (indices % 31 - 15).float() / 32 + torch.arange(32).reshape(
            -1, 1, 1
        ) / 97
        targets = (indices % 29 - 14).float() / 32 - torch.arange(32).reshape(
            -1, 1, 1
        ) / 194
        dataset = TensorDataset(data, targets, torch.arange(32))
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        types = []

        def record_dtype(module, inputs, output):
            types.append(str(output.dtype))

        for layer in base.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.register_forward_hook(record_dtype)
        model, opt, loader, sched = acc.prepare(base, opt_raw, loader, sched_raw)
        assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
        assert (
            type(opt).__name__ == "AcceleratedOptimizer"
            and type(sched).__name__ == "AcceleratedScheduler"
        )
        assert (acc.scaler is not None) == (a.precision == "fp16")
        acc.register_for_checkpointing(progress)
        random.seed(711 + rank)
        np.random.seed(811 + rank)
        torch.manual_seed(911 + rank)
        torch.cuda.manual_seed(1011 + rank)
        traffic = dict(calls=0, bytes=0)
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
            allreduce_hook,
        )

        def count_hook(state, bucket):
            state["calls"] += 1
            state["bytes"] += bucket.buffer().numel() * bucket.buffer().element_size()
            assert bucket.buffer().dtype == torch.float32
            return allreduce_hook(None, bucket)

        model.register_comm_hook(traffic, count_hook)
        elements = sum(v.numel() for v in model.parameters())
        tool = Path(bc250_distributed_identity.__file__).parent
        package = Path(accelerate.__file__).parent
        config = dict(
            precision=a.precision,
            world=2,
            accumulation=4,
            global_batch=8,
            split_batches=True,
            adjust_scheduler=False,
            save_on_each_node=True,
            parameters=elements,
            accelerate=accelerate.__version__,
            numpy=np.__version__,
            workers=0,
            fixture=digest(Path(__file__)),
            native=digest(tool / "torch-identity.json"),
            accelerate_source={
                n: digest(package / n)
                for n in [
                    "accelerator.py",
                    "optimizer.py",
                    "scheduler.py",
                    "state.py",
                    "checkpointing.py",
                    "utils/other.py",
                ]
            },
        )
        result.update(
            rank=rank,
            world=world,
            birth=birth,
            config=config,
            torch=torch.__version__,
            source_sha256=digest(Path(__file__)),
            distributed_type=acc.distributed_type.value,
            backend=acc.state.backend,
        )
        checkpoint = a.seed_directory / "accelerate-checkpoint"
        references = a.seed_directory / "reference"
        checkpoint_state = a.seed_directory / "checkpoint-state.pt"

        def state() -> dict:
            return dict(
                params=[cpu_tree(v) for v in model.parameters()],
                optimizer=cpu_tree(opt.state_dict()),
                scheduler=cpu_tree(sched.state_dict()),
                scaler={} if acc.scaler is None else acc.scaler.state_dict(),
                rng=rng_state(),
                progress=progress.state_dict(),
                accumulation_step=acc.step,
            )

        def collective_check(manifest: dict, label: str) -> None:
            error = None
            try:
                assert manifest["config"] == config and manifest["rank"] == rank
                assert manifest["progress"] == dict(micros=8, updates=2)
                for rec in manifest["files"]:
                    path = checkpoint / rec["name"]
                    assert (
                        path.is_file()
                        and path.stat().st_size == rec["bytes"]
                        and digest(path) == rec["sha256"]
                    )
            except (AssertionError, KeyError, OSError) as exc:
                error = repr(exc)
            errors = [None] * world
            dist.all_gather_object(errors, error)
            if any(v is not None for v in errors):
                raise ValueError(f"Accelerate checkpoint admission refused: {label}")

        if a.action == "seed":
            references.mkdir()
            iterator = iter(loader)
        else:
            manifest = json.loads((a.seed_directory / "complete.json").read_text())
            before = fingerprint(state())
            for label in ["config", "cursor", "file-size", "file-hash"]:
                bad = copy.deepcopy(manifest)
                if rank == 1:
                    if label == "config":
                        bad["config"]["precision"] = "unqualified"
                    elif label == "cursor":
                        bad["progress"]["micros"] = 9
                    elif label == "file-size":
                        bad["files"][0]["bytes"] += 1
                    else:
                        bad["files"][0]["sha256"] = "0" * 64
                try:
                    collective_check(bad, label)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Bad native Accelerate checkpoint admitted")
            assert fingerprint(state()) == before
            collective_check(manifest, "real-checkpoint")
            # Materialize/advance the deterministic loader BEFORE native RNG restoration.
            skipped = acc.skip_first_batches(loader, 8)
            source_iterator = iter(skipped)
            first = next(source_iterator)
            iterator = itertools.chain([first], source_iterator)
            acc.load_state(str(checkpoint))
            expected = torch.load(
                checkpoint_state, map_location="cpu", weights_only=True
            )
            exact(state(), expected)
            assert (
                birth["uuid"] != manifest["birth"]["uuid"]
                and birth["start_ticks"] != manifest["birth"]["start_ticks"]
            )
            result["source_birth"] = manifest["birth"]
            result["checks"].extend(
                [
                    "four-collective-checkpoint-refusals-without-mutation",
                    "native-load_state-restores-initial-model-optimizer-scheduler-scaler-RNG-custom-cursor",
                ]
            )
            result["checkpoint_refusals"] = 4

        expected_loss = None
        for x, target, ids in iterator:
            micro = progress.micros
            window = micro // 4
            position = micro % 4
            assert ids.tolist() == [2 * micro + rank]
            peers = [None] * world
            dist.all_gather_object(peers, ids.tolist())
            assert sorted(v[0] for v in peers) == [2 * micro, 2 * micro + 1]
            if position == 0:
                types.clear()
                window_aug = []
                window_ids = []
                window_loss = torch.zeros((), device=acc.device)
                if a.action == "seed" and window < 2:
                    for p, wanted in zip(model.parameters(), oracle.parameters()):
                        wanted.data.copy_(p.detach().cpu().to(wanted.dtype))
                    oracle.zero_grad(set_to_none=True)
                    cpu_opt.zero_grad(set_to_none=True)
                    expected_loss = torch.zeros(
                        (),
                        dtype=torch.float64 if a.precision == "no" else torch.float32,
                    )
            augmentation = (
                random.random()
                + float(np.random.random())
                + torch.rand(()).item()
                + torch.rand((), device=acc.device).item()
            ) / 1024
            augmentations = [None] * world
            dist.all_gather_object(augmentations, augmentation)
            if a.action == "seed" and window < 2:
                scale = 1 if acc.scaler is None else acc.scaler.get_scale()
                for peer in range(world):
                    i = 2 * micro + peer
                    cpu_x = (
                        data[i : i + 1].double()
                        if a.precision == "no"
                        else data[i : i + 1]
                    )
                    prediction = (
                        oracle(cpu_x)
                        if a.precision == "no"
                        else oracle(cpu_x, oracle_dtype=dtype)
                    )
                    cpu_loss = (
                        prediction
                        - targets[i : i + 1].to(prediction.dtype)
                        * (1 + augmentations[peer])
                    ).square().mean() / 4
                    (cpu_loss * scale).backward()
                    expected_loss += cpu_loss.detach() / world
            before_calls = dict(traffic)
            before_updates = progress.updates
            before_epoch = sched.state_dict()["last_epoch"]
            with acc.accumulate(model):
                if window == 2 and position == 3 and rank == 1:
                    next(model.parameters()).grad.reshape(-1)[0] = float("inf")
                output = model(x)
                assert output.dtype == torch.float32
                loss = (output - target * (1 + augmentation)).square().mean()
                acc.backward(
                    loss
                )  # Accelerate divides by accumulation count; do not divide twice.
                assert acc.sync_gradients == (position == 3)
                window_loss += loss.detach() / 4
                finite = True
                if acc.sync_gradients:
                    acc.unscale_gradients(opt)
                    flags = [None] * world
                    dist.all_gather_object(
                        flags,
                        all(
                            torch.isfinite(v.grad).all().item()
                            for v in model.parameters()
                        ),
                    )
                    assert flags == ([False, False] if window == 2 else [True, True])
                    finite = all(flags)
                    if a.action == "seed" and window < 2:
                        averaged = window_loss.detach().clone()
                        dist.all_reduce(averaged)
                        averaged /= world
                        torch.testing.assert_close(
                            averaged.cpu().double(),
                            expected_loss.double(),
                            atol=3e-6 if a.precision == "no" else 2e-5,
                            rtol=(
                                5e-5
                                if a.precision == "no"
                                else 0.002 if a.precision == "bf16" else 0.0005
                            ),
                        )
                        errors = []
                        for actual, wanted, cpu_param in zip(
                            model.parameters(),
                            oracle.parameters(),
                            cpu_opt_model.parameters(),
                        ):
                            desired = wanted.grad / 2 / scale
                            if a.precision == "no":
                                torch.testing.assert_close(
                                    actual.grad.cpu().double(),
                                    desired,
                                    atol=3e-6,
                                    rtol=5e-4,
                                )
                            else:
                                errors.append(
                                    norm_check(
                                        actual.grad,
                                        desired,
                                        0.02 if a.precision == "bf16" else 0.004,
                                    )
                                )
                            cpu_param.grad = actual.grad.detach().cpu().double().clone()
                        result["gradient_errors"].append(
                            dict(
                                window=window,
                                maximum_relative_L2=max(errors) if errors else 0,
                            )
                        )
                # Exercise wrappers on EVERY micro; they gate optimizer/zero_grad/scheduler internally.
                if not acc.sync_gradients or finite or a.precision == "fp16":
                    opt.step()
                    sched.step()
                assert progress.updates - before_updates == int(
                    acc.sync_gradients and finite
                )
                assert sched.state_dict()["last_epoch"] - before_epoch == int(
                    acc.sync_gradients and finite
                )
                if acc.sync_gradients and a.precision == "fp16":
                    assert acc.optimizer_step_was_skipped == (not finite)
                grads = (
                    [
                        None if v.grad is None else cpu_tree(v.grad)
                        for v in model.parameters()
                    ]
                    if acc.sync_gradients
                    else None
                )
                opt.zero_grad()
                progress.micros += 1
            delta = {k: traffic[k] - before_calls[k] for k in traffic}
            assert delta["bytes"] == (elements * 4 if position == 3 else 0)
            assert bool(delta["calls"]) == (position == 3)
            assert types and set(types) == {str(dtype)}
            window_aug.append(augmentation)
            window_ids.extend(ids.tolist())
            if position == 3:
                assert sched.state_dict()["last_epoch"] == progress.updates
                assert math.isclose(
                    opt.param_groups[0]["lr"],
                    0.001 * 0.8 ** (progress.updates // 2),
                    abs_tol=1e-15,
                    rel_tol=0,
                )
                if a.action == "seed" and window < 2:
                    cpu_opt.step()
                    for actual, wanted in zip(
                        model.parameters(), cpu_opt_model.parameters()
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
                    result["checks"].append(
                        f"Accelerate-global-loss-all-gradients-CPU-AdamW-moments-{window}"
                    )
                value = dict(
                    window=window,
                    loss=cpu_tree(window_loss),
                    grads=grads,
                    augmentation=window_aug,
                    indices=window_ids,
                    finite=finite,
                    state=state(),
                )
                result["transcript"].append(
                    dict(
                        window=window,
                        digest=fingerprint(value),
                        optimizer_updates=progress.updates,
                        finite=finite,
                    )
                )
                if a.action == "seed" and window == 1:
                    initial = state()
                    acc.save_state(str(checkpoint))
                    torch.save(initial, checkpoint_state)
                    files = [
                        dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p))
                        for p in sorted(checkpoint.iterdir())
                        if p.is_file()
                    ]
                    names = {v["name"] for v in files}
                    assert {
                        "model.safetensors",
                        "optimizer.bin",
                        "scheduler.bin",
                        "custom_checkpoint_0.pkl",
                        f"random_states_{rank}.pkl",
                    }.issubset(names)
                    if a.precision == "fp16":
                        assert "scaler.pt" in names
                    manifest = dict(
                        config=config,
                        rank=rank,
                        birth=birth,
                        progress=progress.state_dict(),
                        files=files,
                    )
                    atomic_json(a.seed_directory / "complete.json", manifest)
                    collective_check(manifest, "written-state")
                    result["checks"].append(
                        "native-save_state-complete-model-optimizer-scheduler-RNG-custom-files-on-each-node"
                    )
                elif window >= 2:
                    reference = references / f"window-{window}.pt"
                    if a.action == "seed":
                        torch.save(value, reference)
                    else:
                        expected = torch.load(
                            reference, map_location="cpu", weights_only=True
                        )
                        exact(value, expected)
                        result["checks"].append(
                            f"fresh-native-Accelerate-replay-exact-all-state-{window}"
                        )
        assert progress.state_dict() == dict(micros=16, updates=3)
        assert acc.step == 0
        result["checks"].extend(
            [
                "actual-Accelerate-MULTI_GPU-NCCL-prepared-loader-DDP-optimizer-scheduler",
                "all32-records-once-split-global-batches-and-four-micro-sync-gates",
                "coherent-overflow-skip-update-only-scheduler-and-finite-recovery",
            ]
        )
        result["final_digest"] = fingerprint(state())
        result["final_scaler"] = {} if acc.scaler is None else acc.scaler.state_dict()
        result["final_progress"] = progress.state_dict()
        result["final_scheduler"] = sched.state_dict()
        acc.wait_for_everyone()
        acc.end_training()
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        atomic_json(a.output, result)


if __name__ == "__main__":
    main()
