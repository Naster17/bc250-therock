#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned matched-global-batch transformer training and bounded checkpoint experiment."""
import argparse
from contextlib import nullcontext
import copy
from datetime import timedelta
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

CONFIG = dict(width=512, heads=8, blocks=4, tokens=64, global_batch=8)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["width"]
        self.n1 = torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.proj = torch.nn.Linear(d, d)
        self.n2 = torch.nn.LayerNorm(d)
        self.up = torch.nn.Linear(d, 4 * d)
        self.down = torch.nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = (
            self.qkv(self.n1(x))
            .reshape(b, t, 3, CONFIG["heads"], d // CONFIG["heads"])
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d // CONFIG["heads"])
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, d))
        return x + self.down(torch.nn.functional.gelu(self.up(self.n2(x))))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(".new")
    with temp.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_commit(commit, folder, rank, world):
    if commit is None:
        raise ValueError("Missing checkpoint commit")
    if (
        commit.get("schema") != 1
        or commit.get("config") != CONFIG
        or commit.get("world") != world
    ):
        raise ValueError("Checkpoint schema/config/world mismatch")
    shards = commit.get("shards", [])
    if len(shards) != world or sorted(x.get("rank", -1) for x in shards) != list(
        range(world)
    ):
        raise ValueError("Incomplete or duplicate shard set")
    record = next(x for x in shards if x["rank"] == rank)
    if record["file"] != f"rank-{rank}.pt":
        raise ValueError("Wrong local shard filename")
    path = folder / record["file"]
    if (
        not path.is_file()
        or path.stat().st_size != record["bytes"]
        or sha(path) != record["sha256"]
    ):
        raise ValueError("Missing or damaged local checkpoint shard")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "ddp", "fsdp"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5)
    a = p.parse_args()
    assert 3 <= a.steps <= 12
    result = dict(status="running", mode=a.mode, config=CONFIG, checks=[])
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert len(os.sched_getaffinity(0)) <= 4
        distributed = a.mode != "single"
        if distributed:
            dist.init_process_group(
                "nccl", init_method="env://", timeout=timedelta(seconds=120)
            )
        rank = dist.get_rank() if distributed else 0
        world = dist.get_world_size() if distributed else 1
        assert world == (2 if distributed else 1)
        result.update(
            rank=rank,
            world=world,
            launcher_rank=int(os.environ["RANK"]),
            torch=torch.__version__,
            script_sha256=sha(Path(__file__)),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
        )
        torch.manual_seed(431)
        reference = torch.nn.Sequential(
            *(Block() for _ in range(CONFIG["blocks"]))
        ).double()
        # GPU and independent CPU-double reference start from the same FP32-rounded weights.
        reference = reference.float().double()
        full_elements = sum(x.numel() for x in reference.parameters())

        def make():
            base = copy.deepcopy(reference).float().cuda()
            if a.mode == "ddp":
                model = DDP(
                    base,
                    device_ids=[0],
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                )
            elif a.mode == "fsdp":
                model = FSDP(
                    base,
                    device_id=0,
                    use_orig_params=True,
                    auto_wrap_policy=partial(
                        transformer_auto_wrap_policy, transformer_layer_cls={Block}
                    ),
                    limit_all_gathers=True,
                )
            else:
                model = base
            return model, torch.optim.AdamW(
                model.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
            )

        model, opt = make()
        local_elements = sum(x.numel() for x in model.parameters())
        assert (
            local_elements * world == full_elements
            if a.mode == "fsdp"
            else local_elements == full_elements
        )
        result.update(
            full_parameter_elements=full_elements,
            local_parameter_elements=local_elements,
        )
        shape = (CONFIG["global_batch"], CONFIG["tokens"], CONFIG["width"])
        count = math.prod(shape)
        cpu_data = (
            (torch.arange(count).reshape(shape) % 31 - 15).float() / 32
        ).double()
        cpu_target = (
            (torch.arange(count).reshape(shape) % 29 - 14).float() / 32
        ).double()
        n = CONFIG["global_batch"] // world
        x = cpu_data[rank * n : (rank + 1) * n].float().cuda()
        target = cpu_target[rank * n : (rank + 1) * n].float().cuda()
        refopt = torch.optim.AdamW(
            reference.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )

        def context(grads=False):
            return (
                FSDP.summon_full_params(
                    model, recurse=True, writeback=False, with_grads=grads
                )
                if a.mode == "fsdp"
                else nullcontext()
            )

        def step(index):
            opt.zero_grad(set_to_none=True)
            loss = (model(x) - target * (1 + index / 256)).square().mean()
            loss.backward()
            opt.step()
            return loss

        for index in range(2):
            opt.zero_grad(set_to_none=True)
            refopt.zero_grad(set_to_none=True)
            loss = (model(x) - target * (1 + index / 256)).square().mean()
            loss.backward()
            expected = (
                (reference(cpu_data) - cpu_target * (1 + index / 256)).square().mean()
            )
            expected.backward()
            with context(True):
                for actual, wanted in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(
                        actual.grad.cpu().double(), wanted.grad, atol=3e-6, rtol=5e-4
                    )
            opt.step()
            refopt.step()
            with context():
                for actual, wanted in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(
                        actual.cpu().double(), wanted, atol=4e-6, rtol=5e-5
                    )
            result["checks"].append(f"CPU-double-global-gradient-AdamW-step-{index}")
        # Keep a cheap reconstruction template after releasing the CPU oracle/optimizer.
        initial = {
            k: v.detach().float().clone() for k, v in reference.state_dict().items()
        }
        del refopt, expected, loss
        for parameter in reference.parameters():
            parameter.grad = None
        reference = reference.float()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for i in range(2):
            step(i + 2)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        losses = []
        for i in range(a.steps):
            if distributed:
                dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = step(i + 4)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if distributed:
                both = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
                dist.all_reduce(both, op=dist.ReduceOp.MAX)
                elapsed = both.item()
            samples.append(elapsed)
            losses.append(loss.item())
        memory = dict(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            allocated_after_bytes=torch.cuda.memory_allocated(),
            reserved_after_bytes=torch.cuda.memory_reserved(),
        )
        state_bytes = sum(
            v.numel() * v.element_size()
            for s in opt.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )
        result.update(
            step_seconds=samples,
            median_step_seconds=statistics.median(samples),
            global_tokens_per_second=CONFIG["global_batch"]
            * CONFIG["tokens"]
            / statistics.median(samples),
            losses=losses,
            memory=memory,
            optimizer_tensor_bytes=state_bytes,
            steady_worker_status=Path("/proc/self/status").read_text(),
            host_meminfo=Path("/proc/meminfo").read_text(),
        )
        result["checks"].append("finite-matched-global-batch-timed-updates")
        assert all(math.isfinite(v) for v in losses)
        expected_state_elements = sum(s["exp_avg"].numel() for s in opt.state.values())
        assert expected_state_elements == local_elements
        result["checks"].append("actual-AdamW-state-element-count")

        # Fixed-world local shards + a collectively agreed commit manifest. No shared FS assumed.
        folder = a.output.parent / "checkpoint"
        folder.mkdir()
        part = folder / f"rank-{rank}.pt"
        temp = part.with_suffix(".new")
        saved_parameters = [v.detach().cpu().clone() for v in model.parameters()]
        payload = dict(
            rank=rank,
            world=world,
            mode=a.mode,
            params=saved_parameters,
            optimizer=opt.state_dict(),
        )
        with temp.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(part)
        record = dict(
            rank=rank, file=part.name, bytes=part.stat().st_size, sha256=sha(part)
        )
        shards = [record]
        if distributed:
            shards = [None] * world
            dist.all_gather_object(shards, record)
        commit = dict(schema=1, config=CONFIG, world=world, shards=shards)
        atomic_json(folder / "commit.json", commit)
        commit = json.loads((folder / "commit.json").read_text())
        validate_commit(commit, folder, rank, world)
        result["checks"].append("atomic-parts-and-complete-collective-manifest")
        cases = [
            None,
            {**commit, "world": world + 1},
            {**commit, "config": {}},
            {**commit, "shards": []},
        ]
        bad = copy.deepcopy(commit)
        next(v for v in bad["shards"] if v["rank"] == rank)["sha256"] = "0" * 64
        cases.append(bad)
        for value in cases:
            try:
                validate_commit(value, folder, rank, world)
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed checkpoint accepted")
        # A truncated owned copy must be rejected before torch.load.
        with part.open("rb") as f:
            prefix = f.read(4096)
        short = folder / "damaged-copy"
        short.mkdir()
        (short / part.name).write_bytes(prefix)
        try:
            validate_commit(commit, short, rank, world)
        except ValueError:
            pass
        else:
            raise AssertionError("Truncated shard accepted")
        result["checkpoint_refusals"] = 6
        result["checks"].append("six-preload-checkpoint-refusals")
        index = a.steps + 4
        expected_loss = step(index).detach().cpu()
        wanted_parameters = [v.detach().cpu().clone() for v in model.parameters()]
        wanted_opt = copy.deepcopy(opt.state_dict())
        del payload, opt, model
        gc.collect()
        reference.load_state_dict(initial)
        model, opt = make()
        path = validate_commit(commit, folder, rank, world)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert (
            loaded["rank"] == rank
            and loaded["world"] == world
            and loaded["mode"] == a.mode
        )
        with torch.no_grad():
            for actual, wanted in zip(model.parameters(), loaded["params"]):
                assert actual.shape == wanted.shape
                actual.copy_(wanted.cuda())
        opt.load_state_dict(loaded["optimizer"])
        resumed_loss = step(index).detach().cpu()
        torch.testing.assert_close(resumed_loss, expected_loss, atol=0, rtol=0)
        for actual, wanted in zip(model.parameters(), wanted_parameters):
            torch.testing.assert_close(actual.cpu(), wanted, atol=0, rtol=0)
        current_opt = opt.state_dict()
        assert current_opt["param_groups"] == wanted_opt["param_groups"]
        for key, state in wanted_opt["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(
                    current_opt["state"][key][field].cpu(), value.cpu(), atol=0, rtol=0
                )
        result["checks"].append("exact-next-loss-parameters-AdamW-after-reconstruction")
        # Communication diagnosis is outside the training timing/memory window.
        if distributed:
            timings = []
            for size in [262144, 2097152, 8388608]:
                buffer = torch.empty(size, device="cuda")
                durations = []
                for repeat in range(4):
                    buffer.fill_(rank + 1)
                    dist.barrier()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    dist.all_reduce(buffer)
                    torch.cuda.synchronize()
                    duration = time.perf_counter() - start
                    if repeat:
                        durations.append(duration)
                assert torch.equal(buffer.cpu(), torch.full((size,), 3.0))
                timings.append(dict(bytes=size * 4, seconds=durations))
            result["allreduce_diagnostic"] = timings
            result["checks"].append("full-buffer-1-8-32MiB-AllReduce-references")
            dist.barrier()
            dist.destroy_process_group()
        result["status"] = "passed"
    except BaseException as e:
        result.update(status="failed", error=repr(e))
        raise
    finally:
        atomic_json(a.output, result)


if __name__ == "__main__":
    main()
