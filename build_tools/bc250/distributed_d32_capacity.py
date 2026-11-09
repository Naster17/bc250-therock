#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned 45.9M GPT2: FP32 one-board/DDP/FSDP references and isolated measurements."""
import argparse
from contextlib import nullcontext
import copy
from datetime import timedelta
from functools import partial
import gc
import math
import os
from pathlib import Path
import statistics
import threading
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from capacity_plan_d32 import CONFIG, PARAMETERS, GIB, admit
from distributed_d05_train import atomic_json, sha


def meminfo():
    return {
        s.split(":")[0]: int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if len(s.split()) >= 3 and s.split()[2] == "kB"
    }


def sample_memory():
    host = meminfo()
    status = {
        s.split(":")[0]: int(s.split()[1]) * 1024
        for s in Path("/proc/self/status").read_text().splitlines()
        if s.startswith(("VmRSS:", "VmHWM:"))
    }
    cg = Path("/sys/fs/cgroup")
    out = dict(
        host_available_bytes=host["MemAvailable"],
        host_swap_used_bytes=host["SwapTotal"] - host["SwapFree"],
        **status
    )
    for name in [
        "memory.current",
        "memory.peak",
        "memory.swap.current",
        "memory.swap.peak",
    ]:
        path = cg / name
        if path.is_file():
            out[name] = int(path.read_text())
    if (cg / "memory.events").is_file():
        out["events"] = {
            s.split()[0]: int(s.split()[1])
            for s in (cg / "memory.events").read_text().splitlines()
        }
    return out


class Monitor:
    def __init__(self):
        self.phase = "initialization"
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            self.samples.append(
                dict(phase=self.phase, seconds=time.monotonic(), **sample_memory())
            )
            self.stop.wait(0.2)

    def finish(self):
        self.stop.set()
        self.thread.join()
        return self.samples


def build():
    torch.manual_seed(431)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=CONFIG["vocab"],
            n_positions=CONFIG["tokens"],
            n_ctx=CONFIG["tokens"],
            n_embd=CONFIG["width"],
            n_layer=CONFIG["blocks"],
            n_head=CONFIG["heads"],
            n_inner=4 * CONFIG["width"],
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            attn_implementation="eager",
            pad_token_id=0,
        )
    )


def data(index):
    values = torch.arange(CONFIG["global_batch"] * CONFIG["tokens"]).reshape(
        CONFIG["global_batch"], CONFIG["tokens"]
    )
    ids = ((values * 17 + index * 31) % (CONFIG["vocab"] - 1) + 1).long()
    return ids, ids.clone()


def loss(logits, labels, denominator):
    return (
        torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="sum",
        )
        / denominator
    )


def named(model):
    root = model.module if isinstance(model, DDP) else model
    return {
        n.replace("_fsdp_wrapped_module.", ""): p for n, p in root.named_parameters()
    }


def full(model, grads=False):
    return (
        FSDP.summon_full_params(model, writeback=False, with_grads=grads)
        if isinstance(model, FSDP)
        else nullcontext()
    )


def gpu_backward(model, index, rank, world, traffic):
    ids, labels = data(index)
    local = CONFIG["global_batch"] // world
    start = rank * local
    denominator = CONFIG["global_batch"] * (CONFIG["tokens"] - 1)
    total = torch.zeros((), device="cuda")
    traffic["active"] = True
    try:
        for offset in range(local):
            context = (
                model.no_sync() if world == 2 and offset != local - 1 else nullcontext()
            )
            with context:
                x = ids[start + offset : start + offset + 1].cuda()
                y = labels[start + offset : start + offset + 1].cuda()
                value = loss(model(input_ids=x).logits, y, denominator) * world
                value.backward()
                total += value.detach()
    finally:
        traffic["active"] = False
    return total


def optimizer(model):
    return torch.optim.AdamW(
        model.parameters(), lr=0.0001, eps=1e-5, weight_decay=0.01, foreach=False
    )


def qualify(model, opt, base, rank, world, traffic):
    oracle = copy.deepcopy(base).double()
    refopt = optimizer(oracle)
    ids, labels = data(0)
    expected = loss(
        oracle(input_ids=ids).logits,
        labels,
        CONFIG["global_batch"] * (CONFIG["tokens"] - 1),
    )
    expected.backward()
    opt.zero_grad(set_to_none=True)
    actual_loss = gpu_backward(model, 0, rank, world, traffic)
    if world == 2:
        dist.all_reduce(actual_loss)
        actual_loss /= world
    torch.testing.assert_close(
        actual_loss.cpu().double(), expected.detach(), atol=3e-6, rtol=5e-5
    )
    wanted_named = named(oracle)
    squares = 0.0
    with full(model, True):
        current = named(model)
        assert set(current) == set(wanted_named)
        root = model.module if isinstance(model, DDP) else model
        assert root.get_input_embeddings().weight is root.get_output_embeddings().weight
        for name, p in current.items():
            assert (
                p.grad is not None
                and p.dtype == p.grad.dtype == torch.float32
                and p.is_cuda
            )
            actual = p.grad.detach().cpu().double()
            torch.testing.assert_close(
                actual, wanted_named[name].grad, atol=3e-6, rtol=5e-4
            )
            squares += float(actual.square().sum())
            wanted_named[name].grad = actual
    wanted_norm = math.sqrt(squares)
    norm = (
        model.clip_grad_norm_(0.5)
        if isinstance(model, FSDP)
        else torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    )
    torch.testing.assert_close(
        norm.cpu().double(),
        torch.tensor(wanted_norm, dtype=torch.float64),
        atol=1e-7,
        rtol=1e-5,
    )
    coefficient = min(1.0, 0.5 / (wanted_norm + 1e-6))
    with full(model, True):
        for name, p in named(model).items():
            torch.testing.assert_close(
                p.grad.cpu().double(),
                wanted_named[name].grad * coefficient,
                atol=1e-8,
                rtol=5e-5,
            )
            with torch.no_grad():
                wanted_named[name].copy_(p.cpu())
            wanted_named[name].grad = p.grad.cpu().double().clone()
    opt.step()
    refopt.step()
    allopt = (
        FSDP.full_optim_state_dict(model, opt, rank0_only=False)
        if isinstance(model, FSDP)
        else None
    )
    with full(model):
        current = named(model)
        for name, p in current.items():
            ref = wanted_named[name]
            torch.testing.assert_close(
                p.cpu().double(), ref.detach(), atol=4e-6, rtol=5e-5
            )
            state = allopt["state"][name] if allopt is not None else opt.state[p]
            for key, atol, rtol in [
                ("exp_avg", 3e-7, 5e-4),
                ("exp_avg_sq", 3e-9, 1e-3),
            ]:
                torch.testing.assert_close(
                    state[key].cpu().double(),
                    refopt.state[ref][key],
                    atol=atol,
                    rtol=rtol,
                )
            assert int(state["step"]) == 1
    return dict(
        loss=float(actual_loss),
        CPU_FP64_loss=float(expected.detach()),
        gradient_parameters=len(wanted_named),
        gradient_elements=PARAMETERS,
        root_norm=float(norm),
        CPU_root_norm=wanted_norm,
        clip_coefficient=coefficient,
        ALL_AdamW_parameters_moments_counters=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "ddp", "fsdp"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(status="running", mode=a.mode, config=CONFIG, checks=[])
    monitor = None
    began = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert len(os.sched_getaffinity(0)) <= 4
        world = 1 if a.mode == "single" else 2
        if world == 2:
            dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        rank = dist.get_rank() if world == 2 else 0
        host = meminfo()
        plan = admit(a.mode, host["MemAvailable"], torch.cuda.mem_get_info()[0])
        r.update(
            rank=rank,
            world=world,
            launcher_rank=int(os.environ["RANK"]),
            torch=torch.__version__,
            plan=plan,
            source_sha256=sha(Path(__file__)),
            plan_source_sha256=sha(Path(__file__).with_name("capacity_plan_d32.py")),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
        )
        monitor = Monitor()
        events_before = sample_memory().get("events", {})
        r["cgroup_events_before"] = events_before
        base = build()
        assert sum(p.numel() for p in base.parameters()) == PARAMETERS
        gpu = copy.deepcopy(base).cuda()
        if a.mode == "ddp":
            model = DDP(
                gpu,
                device_ids=[0],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        elif a.mode == "fsdp":
            model = FSDP(
                gpu,
                device_id=0,
                use_orig_params=True,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block}
                ),
                limit_all_gathers=True,
            )
        else:
            model = gpu
        opt = optimizer(model)
        local_elements = sum(p.numel() for p in model.parameters())
        assert (
            local_elements == PARAMETERS // world
            if a.mode == "fsdp"
            else local_elements == PARAMETERS
        )
        traffic = dict(active=False, gathers=[], reductions=[], DDP=[])
        if a.mode == "ddp":

            def hook(state, bucket):
                if state["active"]:
                    state["DDP"].append(
                        bucket.buffer().numel() * bucket.buffer().element_size()
                    )
                return allreduce_hook(None, bucket)

            model.register_comm_hook(traffic, hook)
        native_gather, native_reduce = (
            dist.all_gather_single,
            dist.reduce_scatter_single,
        )

        def gather(output, input, *args, **kwargs):
            if traffic["active"]:
                assert input.dtype == output.dtype == torch.float32
                traffic["gathers"].append(input.numel() * input.element_size())
            return native_gather(output, input, *args, **kwargs)

        def reduce(output, input, *args, **kwargs):
            if traffic["active"]:
                assert input.dtype == output.dtype == torch.float32
                traffic["reductions"].append(input.numel() * input.element_size())
            return native_reduce(output, input, *args, **kwargs)

        dist.all_gather_single, dist.reduce_scatter_single = gather, reduce
        monitor.phase = "CPU-reference-and-audits"
        r["reference"] = qualify(model, opt, base, rank, world, traffic)
        r["checks"].extend(
            [
                "CPU-FP64-global-loss-ALL-gradients",
                "CPU-root-norm-and-ALL-clipped-gradients",
                "CPU-double-ALL-AdamW-parameters-moments-counters",
            ]
        )
        del base, gpu
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        state_elements = sum(s["exp_avg"].numel() for s in opt.state.values())
        state_bytes = sum(
            v.numel() * v.element_size()
            for s in opt.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )
        assert state_elements == local_elements
        r["state"] = dict(
            full_parameter_elements=PARAMETERS,
            local_master_elements=local_elements,
            AdamW_moment_elements_each=state_elements,
            AdamW_state_bytes=state_bytes,
        )

        def step(index):
            opt.zero_grad(set_to_none=True)
            value = gpu_backward(model, index, rank, world, traffic)
            (
                model.clip_grad_norm_(0.5)
                if isinstance(model, FSDP)
                else torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            )
            opt.step()
            return value

        monitor.phase = "warmup"
        training_start = time.perf_counter()
        for index in [1, 2]:
            step(index)
        torch.cuda.synchronize()
        monitor.phase = "measured-training"
        samples = []
        for index in range(3, 8):
            if world == 2:
                dist.barrier()
            torch.cuda.synchronize()
            before = {k: len(traffic[k]) for k in ["gathers", "reductions", "DDP"]}
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            value = step(index)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            metrics = dict(
                seconds_local=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocated_after_bytes=torch.cuda.memory_allocated(),
                reserved_after_bytes=torch.cuda.memory_reserved(),
                device_free_bytes=torch.cuda.mem_get_info()[0],
                **sample_memory()
            )
            if world == 2:
                tensor = torch.tensor(
                    [seconds, float(value)], dtype=torch.float64, device="cuda"
                )
                dist.all_reduce(tensor[:1], op=dist.ReduceOp.MAX)
                dist.all_reduce(tensor[1:])
                metrics.update(
                    seconds_max_rank=float(tensor[0]),
                    global_loss=float(tensor[1]) / world,
                )
            else:
                metrics.update(seconds_max_rank=seconds, global_loss=float(value))
            assert (
                math.isfinite(metrics["global_loss"])
                and metrics["host_available_bytes"] > 3 * GIB
            )
            metrics["traffic_input_bytes"] = {
                k: sum(traffic[k][before[k] :]) for k in before
            }
            metrics["traffic_calls"] = {k: len(traffic[k]) - before[k] for k in before}
            samples.append(metrics)
        r["training_segment_seconds"] = time.perf_counter() - training_start
        r["samples"] = samples
        r["median_update_seconds"] = statistics.median(
            s["seconds_max_rank"] for s in samples
        )
        r["checks"].extend(
            [
                "five-finite-matched-global-batch-updates",
                "actual-master-AdamW-shard-ownership",
                "FP32-training-collective-inputs",
            ]
        )
        if world == 2:
            dist.barrier()
        r.update(status="passed", worker_seconds=time.perf_counter() - began)
        after = sample_memory()
        r["cgroup_events_after"] = after.get("events", {})
        assert all(
            after.get("events", {}).get(k, 0) == events_before.get(k, 0)
            for k in ["oom", "oom_kill"]
        )
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        if monitor is not None:
            r["memory_samples"] = monitor.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
        atomic_json(a.output, r)


if __name__ == "__main__":
    main()
