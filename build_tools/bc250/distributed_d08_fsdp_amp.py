#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D08: classic FSDP mixed precision, full-gradient accumulation and shard overflow."""
import argparse
from contextlib import nullcontext
import copy
from datetime import timedelta
import gc
import hashlib
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from distributed_d05_train import atomic_json, sha, validate_commit
from distributed_d07_amp import (
    CONFIG,
    Network,
    norm_check,
    rounded,
    tree_exact,
    validate_scaler,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], required=True)
    parser.add_argument("--mode", choices=["sync", "no-sync"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    CONFIG.update(
        dtype=args.dtype,
        mode=args.mode,
        strategy="FSDP-FULL_SHARD",
        reduce_dtype="FP32",
    )
    result = dict(
        status="running", config=dict(CONFIG), checks=[], metrics=[], windows=[]
    )
    original_reduce = dist.reduce_scatter_single
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert len(os.sched_getaffinity(0)) <= 4
        dist.init_process_group(
            "nccl", init_method="env://", timeout=timedelta(seconds=120), device_id=0
        )
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 2
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        limit = 0.02 if args.dtype == "bf16" else 0.004
        torch.manual_seed(557)
        template = Network()
        oracle = copy.deepcopy(template)
        opt_oracle = copy.deepcopy(template).double()
        refopt = torch.optim.AdamW(
            opt_oracle.parameters(),
            lr=0.001,
            eps=1e-5,
            weight_decay=0.01,
            foreach=False,
        )
        elements = sum(p.numel() for p in template.parameters())
        traffic = dict(calls=0, bytes=0)
        forward_types = []
        step_calls = [0]

        def counted_reduce(output: torch.Tensor, input: torch.Tensor, *a, **kw):
            traffic["calls"] += 1
            traffic["bytes"] += input.numel() * input.element_size()
            assert input.dtype == torch.float32 and output.dtype == torch.float32
            return original_reduce(output, input, *a, **kw)

        dist.reduce_scatter_single = counted_reduce

        def record_forward(module, inputs) -> None:
            forward_types.append((str(module.weight.dtype), str(inputs[0].dtype)))

        def make():
            base = copy.deepcopy(template).cuda()
            for layer in base.modules():
                if isinstance(layer, torch.nn.Linear):
                    layer.register_forward_pre_hook(record_forward)
            policy = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=torch.float32,
                keep_low_precision_grads=False,
                cast_forward_inputs=False,
                cast_root_forward_inputs=False,
            )
            model = FSDP(
                base,
                device_id=0,
                use_orig_params=True,
                auto_wrap_policy=ModuleWrapPolicy({torch.nn.Linear}),
                mixed_precision=policy,
                limit_all_gathers=True,
            )
            opt = torch.optim.AdamW(
                model.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
            )

            def updated(*_) -> None:
                step_calls[0] += 1

            opt.register_step_post_hook(updated)
            scaler = ShardedGradScaler(
                "cuda", enabled=args.dtype == "fp16", init_scale=16, growth_interval=2
            )
            return model, opt, scaler

        model, opt, scaler = make()
        local_elements = sum(p.numel() for p in model.parameters())
        layouts = [None] * world
        dist.all_gather_object(layouts, [p.numel() for p in model.parameters()])
        assert sum(sum(v) for v in layouts) == elements
        assert local_elements * world == elements
        assert all(
            sum(v[i] for v in layouts) == p.numel()
            for i, p in enumerate(template.parameters())
        )
        result.update(
            rank=rank,
            world=world,
            torch=torch.__version__,
            parameter_elements=elements,
            local_parameter_elements=local_elements,
            layouts=layouts,
            script_sha256=sha(Path(__file__)),
            reused_d07_sha256=sha(Path(__file__).with_name("distributed_d07_amp.py")),
            reused_d05_sha256=sha(Path(__file__).with_name("distributed_d05_train.py")),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
        )
        result["checks"].append("actual-shards-cover-every-original-parameter")
        shape = (CONFIG["global_batch"], CONFIG["tokens"], CONFIG["width"])
        indices = torch.arange(math.prod(shape)).reshape(shape)
        offsets = torch.arange(shape[0]).reshape(-1, 1, 1).float() / 97
        data = (indices % 31 - 15).float() / 32 + offsets
        targets = (indices % 29 - 14).float() / 32 - offsets / 2
        n = shape[0] // world
        x = data[rank * n : (rank + 1) * n].cuda().requires_grad_()
        target = targets[rank * n : (rank + 1) * n].cuda()

        def full(grads: bool = False):
            return FSDP.summon_full_params(
                model, recurse=True, writeback=False, with_grads=grads
            )

        def begin(index: int, fault: str | None = None):
            validate_scaler(scaler)
            opt.zero_grad(set_to_none=True)
            x.grad = None
            total = torch.zeros((), device="cuda")
            records = []
            traces = []
            forward_types.clear()
            for i in range(n):
                before = dict(traffic)
                skip = args.mode == "no-sync" and i + 1 < n
                with model.no_sync() if skip else nullcontext():
                    with torch.autocast("cuda", dtype=dtype):
                        out = model(x[i : i + 1], trace=traces)
                        loss = (
                            out.float() - target[i : i + 1] * (1 + index / 256)
                        ).square().mean() / n
                    assert out.dtype == torch.float32 and loss.dtype == torch.float32
                    scaler.scale(loss).backward()
                torch.cuda.synchronize()
                total += loss.detach()
                delta = {k: traffic[k] - before[k] for k in traffic}
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                grad_elements = sum(g.numel() for g in grads)
                grad_dtypes = sorted({str(g.dtype) for g in grads if g.numel()})
                assert delta["bytes"] == (0 if skip else elements * 4)
                assert delta["calls"] == (0 if skip else 8)
                assert grad_elements == (elements if skip else local_elements)
                assert grad_dtypes == [str(dtype) if skip else "torch.float32"]
                records.append(
                    dict(
                        **delta,
                        gradient_elements=grad_elements,
                        gradient_dtypes=grad_dtypes,
                        allocated_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved(),
                    )
                )
            assert traces and set(traces) == {str(dtype)}
            assert forward_types and {v[0] for v in forward_types} == {str(dtype)}
            assert all(p.dtype == torch.float32 for p in model.parameters())
            if fault is not None and rank == 1:
                next(
                    p.grad
                    for p in model.parameters()
                    if p.grad is not None and p.grad.numel()
                ).reshape(-1)[0] = float(fault)
            # The bad value is deliberately LOCAL after reduce-scatter; the other rank stays finite.
            finite = all(
                torch.isfinite(p.grad).all().item()
                for p in model.parameters()
                if p.grad is not None
            )
            flags = [None] * world
            dist.all_gather_object(flags, finite)
            assert flags == ([True, True] if fault is None else [True, False])
            scaler.unscale_(
                opt
            )  # ShardedGradScaler synchronizes found_inf across distinct local shards.
            result["windows"].append(
                dict(
                    index=index,
                    fault=fault,
                    local_finite_flags=flags,
                    global_finite=all(flags),
                    scale_before=scaler.get_scale(),
                    micros=records,
                )
            )
            return total, all(flags)

        def finish(finite: bool, measure: bool = False) -> None:
            before = step_calls[0]
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            elif finite:
                opt.step()
            assert step_calls[0] - before == int(finite)
            if measure:
                torch.cuda.synchronize()
                result["training_memory"] = dict(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    allocated_after_bytes=torch.cuda.memory_allocated(),
                    reserved_after_bytes=torch.cuda.memory_reserved(),
                    micro_snapshots=result["windows"][-1]["micros"],
                )
            result["windows"][-1]["scale_after"] = scaler.get_scale()
            validate_scaler(scaler)
            with full():
                digest = hashlib.sha256(
                    b"".join(
                        p.detach().cpu().numpy().tobytes() for p in model.parameters()
                    )
                ).hexdigest()
            hashes = [None] * world
            dist.all_gather_object(hashes, digest)
            assert hashes[0] == hashes[1]
            exp_elements = sum(v["exp_avg"].numel() for v in opt.state.values())
            assert exp_elements == local_elements
            result["local_AdamW_exp_avg_elements"] = exp_elements

        for index in range(2):
            with full():
                for actual, wanted in zip(model.parameters(), oracle.parameters()):
                    wanted.data.copy_(actual.detach().cpu().reshape(wanted.shape))
            scale = scaler.get_scale()
            expected_loss = torch.zeros(())
            cpu_x = data.clone().requires_grad_()
            global_grads = [torch.zeros_like(p) for p in oracle.parameters()]
            for cpu_rank in range(world):
                oracle.zero_grad(set_to_none=True)
                for i in range(cpu_rank * n, (cpu_rank + 1) * n):
                    prediction = oracle(cpu_x[i : i + 1], oracle_dtype=dtype)
                    cpu_loss = (
                        prediction - targets[i : i + 1] * (1 + index / 256)
                    ).square().mean() / n
                    (cpu_loss * scale).backward()
                    if args.mode == "no-sync":
                        # Native FSDP accumulates the full flat grad in param_dtype until final reduction.
                        for p in oracle.parameters():
                            p.grad.copy_(rounded(p.grad, dtype))
                    expected_loss += cpu_loss.detach() / world
                for value, p in zip(global_grads, oracle.parameters()):
                    value.add_(p.grad / (world * scale))
            loss, finite = begin(index)
            loss_global = loss.detach().clone()
            dist.all_reduce(loss_global)
            loss_global /= world
            torch.testing.assert_close(
                loss_global.cpu(),
                expected_loss,
                atol=2e-5,
                rtol=0.002 if args.dtype == "bf16" else 0.0005,
            )
            errors = []
            refopt.zero_grad(set_to_none=True)
            with full(True):
                for actual, wanted, cpu_param in zip(
                    model.parameters(), global_grads, opt_oracle.parameters()
                ):
                    errors.append(norm_check(actual.grad, wanted, limit))
                    cpu_param.grad = actual.grad.detach().cpu().double().clone()
            input_error = norm_check(
                x.grad / scale, cpu_x.grad[rank * n : (rank + 1) * n] / scale, limit
            )
            finish(finite)
            refopt.step()
            with full():
                for actual, wanted in zip(model.parameters(), opt_oracle.parameters()):
                    torch.testing.assert_close(
                        actual.detach().cpu().double(), wanted, atol=4e-6, rtol=5e-5
                    )
            for i, (actual, wanted) in enumerate(
                zip(model.parameters(), opt_oracle.parameters())
            ):
                size = actual.numel()
                if not size:
                    assert not opt.state.get(actual, {})
                    continue
                offset = sum(layouts[j][i] for j in range(rank))
                for field, atol, rtol in [
                    ("exp_avg", 3e-7, 5e-4),
                    ("exp_avg_sq", 3e-9, 1e-3),
                ]:
                    expected = refopt.state[wanted][field].reshape(-1)[
                        offset : offset + size
                    ]
                    torch.testing.assert_close(
                        opt.state[actual][field].cpu().double().reshape(-1),
                        expected,
                        atol=atol,
                        rtol=rtol,
                    )
                tree_exact(opt.state[actual]["step"], refopt.state[wanted]["step"])
            result["metrics"].append(
                dict(
                    index=index,
                    maximum_parameter_gradient_relative_L2=max(errors),
                    input_gradient_relative_L2=input_error,
                )
            )
            result["checks"].extend(
                [
                    f"CPU-rounding-FP64-loss-all-gradients-{index}",
                    f"CPU-FP64-AdamW-owned-shards-{index}",
                ]
            )
        result["checks"].append(
            "low-precision-full-forward-and-FP32-master-gradient-shards"
        )
        # Isolate a training-only allocator window; exclude CPU oracle/summon/checkpoint work.
        opt.zero_grad(set_to_none=True)
        x.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result["memory_start"] = dict(
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
        )
        loss, finite = begin(2)
        finish(finite, measure=True)
        result["checks"].append(
            "isolated-training-window-and-full-vs-sharded-grad-counts"
        )
        if scaler.is_enabled():
            saved = scaler.state_dict()
            assert saved["scale"] == 32 and saved["_growth_tracker"] == 1
            for field in ["scale", "_growth_tracker"]:
                wrong = dict(saved)
                wrong[field] = (
                    saved[field] * 2 if field == "scale" else saved[field] + 1
                )
                if rank == 1:
                    scaler.load_state_dict(wrong)
                try:
                    validate_scaler(scaler)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Divergent shard scaler state admitted")
                scaler.load_state_dict(saved)
            result["checks"].append("two-collective-scaler-mismatch-refusals")
        for index, fault in [(3, "inf"), (4, "nan")]:
            saved_params = [p.detach().cpu().clone() for p in model.parameters()]
            saved_opt = copy.deepcopy(opt.state_dict())
            scale = scaler.get_scale()
            _, finite = begin(index, fault)
            finish(finite)
            for actual, wanted in zip(model.parameters(), saved_params):
                tree_exact(actual, wanted)
            tree_exact(opt.state_dict(), saved_opt)
            if scaler.is_enabled():
                assert (
                    scaler.get_scale() == scale / 2
                    and scaler.state_dict()["_growth_tracker"] == 0
                )
            result["checks"].append(f"rank1-local-shard-{fault}-both-skip-exact-state")
        loss, finite = begin(5)
        finish(finite)
        result["checks"].append("finite-recovery-after-local-shard-nonfinite")
        folder = args.output.parent / "checkpoint"
        folder.mkdir()
        part = folder / f"rank-{rank}.pt"
        payload = dict(
            params=[p.detach().cpu().clone() for p in model.parameters()],
            optimizer=opt.state_dict(),
            scaler=scaler.state_dict(),
            rank=rank,
            world=world,
            config=dict(CONFIG),
            next_index=6,
        )
        temporary = part.with_suffix(".new")
        with temporary.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(part)
        record = dict(
            rank=rank, file=part.name, bytes=part.stat().st_size, sha256=sha(part)
        )
        shards = [None] * world
        dist.all_gather_object(shards, record)
        commit = dict(schema=1, config=dict(CONFIG), world=world, shards=shards)
        atomic_json(folder / "commit.json", commit)
        validate_commit(commit, folder, rank, world)
        refused = [
            None,
            {**commit, "world": 3},
            {**commit, "config": {}},
            {**commit, "shards": []},
        ]
        bad = copy.deepcopy(commit)
        bad["shards"][rank]["sha256"] = "0" * 64
        refused.append(bad)
        for value in refused:
            try:
                validate_commit(value, folder, rank, world)
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed checkpoint admitted")
        short = folder / "damaged-copy"
        short.mkdir()
        with part.open("rb") as f:
            (short / part.name).write_bytes(f.read(4096))
        try:
            validate_commit(commit, short, rank, world)
        except ValueError:
            pass
        else:
            raise AssertionError("Truncated checkpoint admitted")
        result["checkpoint_refusals"] = 6
        result["checks"].append("atomic-collective-checkpoint-and-six-preload-refusals")
        expected_loss, finite = begin(6)
        finish(finite)
        expected_loss = expected_loss.detach().cpu()
        wanted_params = [p.detach().cpu().clone() for p in model.parameters()]
        wanted_opt = copy.deepcopy(opt.state_dict())
        wanted_scaler = scaler.state_dict()
        model, opt, scaler = make()
        loaded = torch.load(
            validate_commit(commit, folder, rank, world),
            map_location="cpu",
            weights_only=True,
        )
        assert (
            loaded["rank"] == rank
            and loaded["world"] == world
            and loaded["config"] == CONFIG
            and loaded["next_index"] == 6
        )
        with torch.no_grad():
            for actual, wanted in zip(model.parameters(), loaded["params"]):
                actual.copy_(wanted.cuda())
        opt.load_state_dict(loaded["optimizer"])
        scaler.load_state_dict(loaded["scaler"])
        resumed_loss, finite = begin(loaded["next_index"])
        finish(finite)
        tree_exact(resumed_loss, expected_loss)
        for actual, wanted in zip(model.parameters(), wanted_params):
            tree_exact(actual, wanted)
        tree_exact(opt.state_dict(), wanted_opt)
        assert scaler.state_dict() == wanted_scaler
        result["checks"].append(
            "exact-next-loss-parameters-AdamW-scaler-after-reconstruction"
        )
        result["worker_status"] = Path("/proc/self/status").read_text()
        result["host_meminfo"] = Path("/proc/meminfo").read_text()
        dist.barrier()
        dist.destroy_process_group()
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        dist.reduce_scatter_single = original_reduce
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
