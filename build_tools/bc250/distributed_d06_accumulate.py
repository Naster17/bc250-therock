#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D06: matched-global-batch gradient accumulation and counted DDP communication."""
import argparse
from contextlib import nullcontext
import copy
from datetime import timedelta
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
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from distributed_d05_train import Block, CONFIG, sha, atomic_json, validate_commit


def counted_allreduce(state, bucket):
    """Count actual DDP bucket reductions, delegating to Torch's averaging hook."""
    state["calls"] += 1
    state["bytes"] += bucket.buffer().numel() * bucket.buffer().element_size()
    return allreduce_hook(None, bucket)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "sync", "no-sync"], required=True)
    p.add_argument("--case", choices=["small", "larger"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5)
    a = p.parse_args()
    assert 3 <= a.steps <= 12
    CONFIG.update(
        global_batch=8 if a.case == "small" else 32,
        micro_batch=1 if a.case == "small" else 2,
    )
    result = dict(status="running", mode=a.mode, config=dict(CONFIG), checks=[])
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
        n = CONFIG["global_batch"] // world
        micro = CONFIG["micro_batch"]
        assert n % micro == 0
        accumulation = n // micro
        result.update(
            rank=rank,
            world=world,
            accumulation=accumulation,
            launcher_rank=int(os.environ["RANK"]),
            torch=torch.__version__,
            script_sha256=sha(Path(__file__)),
            reused_d05_sha256=sha(Path(__file__).with_name("distributed_d05_train.py")),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
        )
        torch.manual_seed(431)
        reference = (
            torch.nn.Sequential(*(Block() for _ in range(CONFIG["blocks"])))
            .float()
            .double()
        )
        full_elements = sum(x.numel() for x in reference.parameters())
        traffic = dict(calls=0, bytes=0)

        def make():
            base = copy.deepcopy(reference).float().cuda()
            if distributed:
                model = DDP(
                    base,
                    device_ids=[0],
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                )
                model.register_comm_hook(traffic, counted_allreduce)
            else:
                model = base
            return model, torch.optim.AdamW(
                model.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
            )

        model, opt = make()
        local_elements = sum(x.numel() for x in model.parameters())
        assert local_elements == full_elements
        result.update(
            full_parameter_elements=full_elements,
            local_parameter_elements=local_elements,
            gradient_bytes=full_elements * 4,
        )
        shape = (CONFIG["global_batch"], CONFIG["tokens"], CONFIG["width"])
        count = math.prod(shape)
        # Distinct samples/ranks: unlike the original periodic fixture, add per-example offsets.
        sample = torch.arange(shape[0]).reshape(-1, 1, 1).double() / 97
        cpu_data = (
            (torch.arange(count).reshape(shape) % 31 - 15).float() / 32
        ).double() + sample
        cpu_target = (
            (torch.arange(count).reshape(shape) % 29 - 14).float() / 32
        ).double() - sample / 2
        # CPU oracle uses the identical FP32-rounded inputs, independently chunked in groups of four.
        cpu_data = cpu_data.float().double()
        cpu_target = cpu_target.float().double()
        x = cpu_data[rank * n : (rank + 1) * n].float().cuda()
        target = cpu_target[rank * n : (rank + 1) * n].float().cuda()
        refopt = torch.optim.AdamW(
            reference.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )
        traffic_steps = []

        def backward(index):
            total = torch.zeros((), device="cuda")
            per_micro = []
            for start in range(0, n, micro):
                before = dict(traffic)
                # no_sync must enclose BOTH forward and backward, with a final synchronized microbatch.
                skip = distributed and a.mode == "no-sync" and start + micro < n
                with model.no_sync() if skip else nullcontext():
                    loss = (
                        model(x[start : start + micro])
                        - target[start : start + micro] * (1 + index / 256)
                    ).square().mean() / accumulation
                    loss.backward()
                total += loss.detach()
                delta = {k: traffic[k] - before[k] for k in traffic}
                expected_bytes = full_elements * 4 if distributed and not skip else 0
                assert delta["bytes"] == expected_bytes, (
                    index,
                    start,
                    delta,
                    expected_bytes,
                )
                assert (delta["calls"] > 0) == bool(expected_bytes)
                per_micro.append(delta)
            traffic_steps.append(
                dict(
                    index=index,
                    microbatches=per_micro,
                    calls=sum(v["calls"] for v in per_micro),
                    bytes=sum(v["bytes"] for v in per_micro),
                )
            )
            return total

        def step(index):
            opt.zero_grad(set_to_none=True)
            loss = backward(index)
            opt.step()
            return loss

        for index in range(2):
            opt.zero_grad(set_to_none=True)
            refopt.zero_grad(set_to_none=True)
            loss = backward(index)
            expected = torch.zeros((), dtype=torch.float64)
            for start in range(0, CONFIG["global_batch"], 4):
                ref_loss = (
                    reference(cpu_data[start : start + 4])
                    - cpu_target[start : start + 4] * (1 + index / 256)
                ).square().mean() * (4 / CONFIG["global_batch"])
                ref_loss.backward()
                expected += ref_loss.detach()
            reported_loss = loss.detach().clone()
            if distributed:
                dist.all_reduce(reported_loss)
                reported_loss /= world
            torch.testing.assert_close(
                reported_loss.cpu().double(), expected, atol=3e-6, rtol=5e-5
            )
            for actual, wanted in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(
                    actual.grad.cpu().double(), wanted.grad, atol=3e-6, rtol=5e-4
                )
            opt.step()
            refopt.step()
            for actual, wanted in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(
                    actual.cpu().double(), wanted, atol=4e-6, rtol=5e-5
                )
                for field, atol, rtol in [
                    ("exp_avg", 3e-7, 5e-4),
                    ("exp_avg_sq", 3e-9, 1e-3),
                ]:
                    torch.testing.assert_close(
                        opt.state[actual][field].cpu().double(),
                        refopt.state[wanted][field],
                        atol=atol,
                        rtol=rtol,
                    )
                torch.testing.assert_close(
                    opt.state[actual]["step"],
                    refopt.state[wanted]["step"],
                    atol=0,
                    rtol=0,
                )
            result["checks"].append(
                f"CPU-double-loss-all-gradients-parameters-AdamW-state-{index}"
            )
        initial = {
            k: v.detach().float().clone() for k, v in reference.state_dict().items()
        }
        del refopt, expected, loss, ref_loss
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
        samples, losses = [], []
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
        result.update(
            step_seconds=samples,
            median_step_seconds=statistics.median(samples),
            global_tokens_per_second=CONFIG["global_batch"]
            * CONFIG["tokens"]
            / statistics.median(samples),
            losses=losses,
            memory=memory,
            steady_worker_status=Path("/proc/self/status").read_text(),
            host_meminfo=Path("/proc/meminfo").read_text(),
        )
        assert all(math.isfinite(v) for v in losses)
        result["checks"].append("finite-matched-global-batch-timed-updates")
        assert sum(s["exp_avg"].numel() for s in opt.state.values()) == local_elements
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
        result["traffic_steps"] = traffic_steps
        result["checks"].append("every-microbatch-bucket-call-and-byte-contract")
        if distributed:
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
