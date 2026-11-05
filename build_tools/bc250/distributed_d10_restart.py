#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned fresh-process DDP/FSDP replay with dropout, scheduler and five RNG streams."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from distributed_d05_train import atomic_json
from distributed_d07_amp import Network
from distributed_d10_checkpoint import canonical, digest, local_part
import bc250_distributed_identity


class StochasticNetwork(Network):
    def forward(self, x):
        return torch.nn.functional.dropout(super().forward(x), p=0.125, training=True)


class Cursor:
    def __init__(self, seed: int):
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.offset = 0, 0
        self.order = torch.randperm(32, generator=self.generator)

    def next(self) -> torch.Tensor:
        if self.offset == 32:
            self.epoch += 1
            self.offset = 0
            self.order = torch.randperm(32, generator=self.generator)
        value = self.order[self.offset : self.offset + 8].clone()
        self.offset += 8
        return value

    def state_dict(self) -> dict:
        return dict(
            epoch=self.epoch,
            offset=self.offset,
            order=self.order.clone(),
            generator=self.generator.get_state(),
        )

    def load(self, state: dict) -> None:
        self.epoch, self.offset = state["epoch"], state["offset"]
        self.order = state["order"].clone()
        self.generator.set_state(state["generator"])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def exact(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual.cpu(), expected.cpu(), atol=0, rtol=0, equal_nan=False
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            exact(a, e)
    else:
        assert actual == expected, (actual, expected)


def fingerprint(value) -> str:
    h = hashlib.sha256()

    def add(v):
        h.update(type(v).__name__.encode())
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().contiguous()
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=str):
                add(key)
                add(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v:
                add(item)
        else:
            h.update(repr(v).encode())

    add(value)
    return h.hexdigest()


def rng_state() -> dict:
    name, keys, pos, gauss, cached = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=dict(name=name, keys=keys.tolist(), pos=pos, gauss=gauss, cached=cached),
        torch_cpu=torch.get_rng_state(),
        torch_gpu=torch.cuda.get_rng_state(0),
    )


def restore_rng(value: dict) -> None:
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state(
        (
            n["name"],
            np.asarray(n["keys"], dtype=np.uint32),
            n["pos"],
            n["gauss"],
            n["cached"],
        )
    )
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state(value["torch_gpu"], 0)


def atomic_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".new")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def agree(error: str | None, label: str, identity: str | None = None) -> None:
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, dict(error=error, label=label, identity=identity))
    if (
        any(v["error"] is not None for v in states)
        or len({v["label"] for v in states}) != 1
        or len({v["identity"] for v in states}) != 1
    ):
        raise ValueError(f"Collective checkpoint refusal {label}: {states}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["ddp", "fsdp"], required=True)
    parser.add_argument("--action", choices=["seed", "resume"], required=True)
    parser.add_argument("--seed-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    result = dict(
        status="running", case=a.case, action=a.action, checks=[], transcript=[]
    )
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert len(os.sched_getaffinity(0)) <= 4
        torch.use_deterministic_algorithms(True)
        dist.init_process_group(
            "nccl", init_method="env://", timeout=timedelta(seconds=120), device_id=0
        )
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 2
        birth = dict(
            uuid=uuid.uuid4().hex,
            pid=os.getpid(),
            pid_namespace=os.stat("/proc/self/ns/pid").st_ino,
            start_ticks=Path("/proc/self/stat")
            .read_text()
            .split(") ", 1)[1]
            .split()[19],
        )
        config = dict(
            case=a.case,
            dtype="fp32" if a.case == "ddp" else "fp16",
            world=2,
            records=32,
            global_batch=8,
            model=(
                "Bert22561-dropout.1" if a.case == "ddp" else "MLP1051648-dropout.125"
            ),
            accumulation=2 if a.case == "ddp" else 4,
            optimizer="AdamW-.001-eps1e-5-wd.01-foreachFalse",
            scheduler="StepLR-2-.8-update-only",
            deterministic=True,
            checkpoint_next_attempt=3,
        )
        tools = Path(bc250_distributed_identity.__file__).parent
        sources = [
            Path(__file__),
            Path(__file__).with_name("distributed_d10_checkpoint.py"),
            Path(__file__).with_name("distributed_d07_amp.py"),
            Path(__file__).with_name("distributed_d05_train.py"),
        ]
        cohort = canonical(
            dict(
                source_sha256={p.name: digest(p) for p in sources},
                native_pin=digest(tools / "torch-identity.json"),
                agent=digest(tools / "torchrun-fix.json"),
                worker=digest(tools / "bc250_distributed_worker.py"),
                torch=torch.__version__,
            )
        )
        torch.manual_seed(773 if a.action == "seed" else 1773 + rank)
        if a.case == "ddp":
            from transformers import BertConfig, BertForMaskedLM

            cfg = BertConfig(
                vocab_size=97,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=32,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                pad_token_id=0,
                attn_implementation="eager",
            )
            model = DDP(
                BertForMaskedLM(cfg).cuda(), device_ids=[0], forward_sync_buffers=False
            )
            scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                keep_low_precision_grads=False,
                cast_forward_inputs=False,
                cast_root_forward_inputs=False,
            )
            model = FSDP(
                StochasticNetwork().cuda(),
                device_id=0,
                use_orig_params=True,
                auto_wrap_policy=ModuleWrapPolicy({torch.nn.Linear}),
                mixed_precision=policy,
                limit_all_gathers=True,
            )
            scaler = ShardedGradScaler("cuda", init_scale=16, growth_interval=2)
        opt = torch.optim.AdamW(
            model.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.8)
        updates = [0]

        def updated(*_) -> None:
            updates[0] += 1

        opt.register_step_post_hook(updated)
        layouts = [None] * world
        dist.all_gather_object(layouts, [list(p.shape) for p in model.parameters()])
        config["layouts"] = layouts
        config["numpy"] = np.__version__
        if a.case == "ddp":
            import transformers

            config["transformers"] = transformers.__version__
        cursor = Cursor(911 if a.action == "seed" else 1911 + rank)
        random.seed(700 + rank + (1000 if a.action == "resume" else 0))
        np.random.seed(800 + rank + (1000 if a.action == "resume" else 0))
        torch.manual_seed(900 + rank + (1000 if a.action == "resume" else 0))
        torch.cuda.manual_seed(1000 + rank + (1000 if a.action == "resume" else 0))
        result.update(
            rank=rank,
            world=world,
            birth=birth,
            config=config,
            cohort=cohort,
            torch=torch.__version__,
            local_parameter_elements=sum(p.numel() for p in model.parameters()),
            source_sha256=digest(Path(__file__)),
        )

        def state() -> dict:
            return dict(
                params=[cpu_tree(p) for p in model.parameters()],
                buffers={k: cpu_tree(v) for k, v in model.named_buffers()},
                optimizer=cpu_tree(opt.state_dict()),
                scheduler=cpu_tree(scheduler.state_dict()),
                scaler=scaler.state_dict(),
                rng=rng_state(),
                cursor=cursor.state_dict(),
                optimizer_updates=updates[0],
            )

        def validate_payload(payload: dict) -> None:
            if (
                payload.get("rank") != rank
                or payload.get("world") != world
                or payload.get("config") != config
                or payload.get("cohort") != cohort
                or payload.get("next_attempt") != 3
            ):
                raise ValueError("Loaded checkpoint metadata mismatch")
            s = payload["state"]
            c = s["cursor"]
            if (
                c["epoch"] != 0
                or c["offset"] != 24
                or sorted(c["order"].tolist()) != list(range(32))
                or s["optimizer_updates"] != 3
            ):
                raise ValueError("Loaded checkpoint cursor/update boundary mismatch")
            if s["scheduler"]["last_epoch"] != 3 or len(s["params"]) != len(
                list(model.parameters())
            ):
                raise ValueError(
                    "Loaded checkpoint scheduler/parameter layout mismatch"
                )
            if any(
                saved.shape != actual.shape or saved.dtype != actual.dtype
                for saved, actual in zip(s["params"], model.parameters())
            ):
                raise ValueError("Loaded checkpoint tensor layout mismatch")
            if a.case == "fsdp" and (
                s["scaler"].get("scale") != 32
                or s["scaler"].get("_growth_tracker") != 1
            ):
                raise ValueError("Loaded shard scaler boundary mismatch")
            if a.case == "ddp" and s["scaler"] != {}:
                raise ValueError("Unexpected FP32 scaler state")

        def checked_part(commit, folder, label):
            error = None
            path = None
            try:
                path = local_part(commit, folder, config, cohort, rank, world, 3)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                error = repr(exc)
            agree(error, label, canonical(commit))
            return path

        def step(attempt: int) -> dict:
            # Exercise Python, NumPy and default CPU RNG independently of the shuffled cursor generator.
            augment = [
                random.random(),
                float(np.random.random()),
                torch.rand(()).item(),
            ]
            indices = cursor.next()
            local = indices[rank * 4 : (rank + 1) * 4]
            opt.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device="cuda")
            count = 2 if a.case == "ddp" else 4
            for micro in range(count):
                if (
                    a.case == "ddp"
                    and attempt == 4
                    and rank == 1
                    and micro == count - 1
                ):
                    next(model.parameters()).grad.reshape(-1)[0] = float("inf")
                batch = local[micro * (4 // count) : (micro + 1) * (4 // count)]
                with model.no_sync() if micro + 1 < count else nullcontext():
                    if a.case == "ddp":
                        ids = (
                            batch[:, None] * 7
                            + torch.arange(12)[None, :] * 3
                            + int(augment[1] * 7)
                        ) % 96 + 1
                        labels = (ids * 3 + int(augment[0] * 11) + attempt) % 97
                        loss = (
                            model(input_ids=ids.cuda(), labels=labels.cuda()).loss
                            / count
                        )
                    else:
                        raw = batch[:, None, None] * 13 + torch.arange(
                            16 * 256
                        ).reshape(1, 16, 256)
                        x = ((raw % 31 - 15).float() / 32 + augment[2] / 64).cuda()
                        target = ((raw % 29 - 14).float() / 32 + augment[0] / 64).cuda()
                        with torch.autocast("cuda", dtype=torch.float16):
                            loss = (model(x).float() - target).square().mean() / count
                    scaler.scale(loss).backward()
                loss_sum += loss.detach()
            if a.case == "fsdp" and attempt == 4 and rank == 1:
                next(
                    p.grad
                    for p in model.parameters()
                    if p.grad is not None and p.grad.numel()
                ).reshape(-1)[0] = float("inf")
            scaler.unscale_(opt)
            flags = [None] * world
            dist.all_gather_object(
                flags,
                all(
                    torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                    if p.grad is not None
                ),
            )
            finite = all(flags)
            assert finite == (attempt != 4)
            before = updates[0]
            scale_before = scaler.get_scale()
            lr_before = opt.param_groups[0]["lr"]
            grads = [
                None if p.grad is None else cpu_tree(p.grad) for p in model.parameters()
            ]
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            elif finite:
                opt.step()
            assert updates[0] - before == int(finite)
            if finite:
                scheduler.step()
            expected_lr = 0.001 * 0.8 ** (updates[0] // 2)
            assert math.isclose(
                opt.param_groups[0]["lr"], expected_lr, rel_tol=0, abs_tol=1e-15
            )
            assert all(v["step"].item() == updates[0] for v in opt.state.values())
            torch.cuda.synchronize()
            return dict(
                attempt=attempt,
                local_loss=cpu_tree(loss_sum),
                indices=indices,
                augment=augment,
                grads=grads,
                finite=finite,
                local_finite_flags=flags,
                scale_before=scale_before,
                scale_after=scaler.get_scale(),
                lr_before=lr_before,
                lr_after=opt.param_groups[0]["lr"],
                state=state(),
            )

        checkpoint = a.seed_directory / "checkpoint"
        references = a.seed_directory / "reference"
        if a.action == "seed":
            checkpoint.mkdir()
            references.mkdir()
            for attempt in range(3):
                value = step(attempt)
                result["transcript"].append(
                    dict(
                        attempt=attempt,
                        digest=fingerprint(value),
                        optimizer_updates=updates[0],
                    )
                )
            payload = dict(
                rank=rank,
                world=world,
                config=config,
                cohort=cohort,
                next_attempt=3,
                birth=birth,
                state=state(),
            )
            part = checkpoint / f"rank-{rank}.pt"
            atomic_torch(part, payload)
            record = dict(
                rank=rank,
                file=part.name,
                bytes=part.stat().st_size,
                sha256=digest(part),
            )
            parts = [None] * world
            dist.all_gather_object(parts, record)
            commit = dict(
                schema=1,
                config=config,
                cohort=cohort,
                world=world,
                next_attempt=3,
                parts=parts,
            )
            atomic_json(checkpoint / "commit.json", commit)
            checked_part(commit, checkpoint, "written-generation")
            reference_records = []
            for attempt in range(3, 6):
                value = step(attempt)
                path = references / f"step-{attempt}.pt"
                atomic_torch(path, value)
                reference_records.append(
                    dict(
                        attempt=attempt,
                        file=path.name,
                        bytes=path.stat().st_size,
                        sha256=digest(path),
                    )
                )
                result["transcript"].append(
                    dict(
                        attempt=attempt,
                        digest=fingerprint(value),
                        optimizer_updates=updates[0],
                        finite=value["finite"],
                    )
                )
            atomic_json(
                references / "manifest.json",
                dict(
                    rank=rank,
                    world=world,
                    config=config,
                    cohort=cohort,
                    birth=birth,
                    records=reference_records,
                ),
            )
            result["checks"].extend(
                [
                    "three-complete-updates-and-atomic-collective-checkpoint",
                    "three-uninterrupted-stochastic-reference-updates",
                    "scheduler-skips-overflow-and-crosses-epoch",
                ]
            )
        else:
            commit = json.loads((checkpoint / "commit.json").read_text())
            before = fingerprint(state())
            # All ranks stay alive and reject together when only rank1's local admission fails.
            cases = [
                "missing-commit",
                "wrong-world",
                "wrong-config",
                "wrong-cohort",
                "duplicate-rank",
                "wrong-step",
                "wrong-digest",
                "truncated-part",
            ]
            for name in cases:
                trial = copy.deepcopy(commit)
                folder = checkpoint
                if rank == 1:
                    if name == "missing-commit":
                        trial = None
                    elif name == "wrong-world":
                        trial["world"] = 3
                    elif name == "wrong-config":
                        trial["config"]["dtype"] = "unqualified"
                    elif name == "wrong-cohort":
                        trial["cohort"] = "0" * 64
                    elif name == "duplicate-rank":
                        trial["parts"][0]["rank"] = 1
                    elif name == "wrong-step":
                        trial["next_attempt"] = 4
                    elif name == "wrong-digest":
                        trial["parts"][1]["sha256"] = "0" * 64
                    else:
                        folder = a.output.parent / "truncated-owned-copy"
                        folder.mkdir()
                        with (checkpoint / f"rank-{rank}.pt").open("rb") as f:
                            (folder / f"rank-{rank}.pt").write_bytes(f.read(4096))
                try:
                    checked_part(trial, folder, name)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid checkpoint admitted")
            path = checked_part(commit, checkpoint, "real-generation")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for name in ["loaded-rank", "loaded-cursor"]:
                bad = copy.deepcopy(payload)
                if rank == 1:
                    if name == "loaded-rank":
                        bad["rank"] = 0
                    else:
                        bad["state"]["cursor"]["offset"] = 999
                error = None
                try:
                    validate_payload(bad)
                except ValueError as exc:
                    error = repr(exc)
                try:
                    agree(error, name)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Bad loaded metadata admitted")
            assert fingerprint(state()) == before
            error = None
            try:
                validate_payload(payload)
            except ValueError as exc:
                error = repr(exc)
            agree(error, "real-payload")
            reference = json.loads((references / "manifest.json").read_text())
            error = None
            try:
                assert (
                    reference["config"] == config
                    and reference["cohort"] == cohort
                    and reference["rank"] == rank
                    and reference["birth"] == payload["birth"]
                )
                for rec in reference["records"]:
                    p = references / rec["file"]
                    assert (
                        p.stat().st_size == rec["bytes"] and digest(p) == rec["sha256"]
                    )
            except (OSError, AssertionError) as exc:
                error = repr(exc)
            agree(error, "uninterrupted-reference-files")
            assert (
                birth["uuid"] != payload["birth"]["uuid"]
                and birth["pid_namespace"] != payload["birth"]["pid_namespace"]
            )
            result["source_birth"] = payload["birth"]
            s = payload["state"]
            with torch.no_grad():
                for p, wanted in zip(model.parameters(), s["params"]):
                    p.copy_(wanted.to(device=p.device, dtype=p.dtype))
                buffers = dict(model.named_buffers())
                assert buffers.keys() == s["buffers"].keys()
                for k, v in s["buffers"].items():
                    buffers[k].copy_(v.to(buffers[k].device))
            opt.load_state_dict(s["optimizer"])
            scheduler.load_state_dict(s["scheduler"])
            scaler.load_state_dict(s["scaler"])
            updates[0] = s["optimizer_updates"]
            cursor.load(s["cursor"])
            restore_rng(s["rng"])
            exact(state(), s)
            result["checks"].extend(
                [
                    "eight-collective-preload-refusals-no-state-mutation",
                    "two-collective-loaded-metadata-refusals",
                    "fresh-process-full-initial-state-restored",
                ]
            )
            for attempt in range(3, 6):
                actual = step(attempt)
                wanted = torch.load(
                    references / f"step-{attempt}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                exact(actual, wanted)
                result["transcript"].append(
                    dict(
                        attempt=attempt,
                        digest=fingerprint(actual),
                        optimizer_updates=updates[0],
                        finite=actual["finite"],
                    )
                )
                result["checks"].append(
                    f"exact-uninterrupted-loss-gradients-params-optimizer-scaler-scheduler-RNG-cursor-{attempt}"
                )
            result["checkpoint_refusals"] = 10
        assert cursor.epoch == 1 and cursor.offset == 16 and updates[0] == 5
        result["checks"].append("final-epoch-one-offset16-five-actual-updates")
        result["final_state_digest"] = fingerprint(state())
        result["final_scaler"] = scaler.state_dict()
        result["worker_status"] = Path("/proc/self/status").read_text()
        dist.barrier()
        dist.destroy_process_group()
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        atomic_json(a.output, result)


if __name__ == "__main__":
    main()
