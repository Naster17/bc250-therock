#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D07 bounded AMP DDP accumulation, independent rounding oracle and scaler replay."""
import argparse
from contextlib import nullcontext
import copy
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

from distributed_d05_train import CONFIG, atomic_json, sha, validate_commit

CONFIG.clear()
CONFIG.update(width=256, hidden=512, blocks=4, tokens=16, global_batch=8, micro_batch=1)


def rounded(value, dtype):
    return value.to(dtype).float()


class OracleLinear(torch.autograd.Function):
    """CPU FP64 matrix arithmetic with explicit AMP operand/output/gradient rounding."""

    @staticmethod
    def forward(ctx, x, weight, bias, dtype):
        qx, qw, qb = [rounded(v, dtype).double() for v in [x, weight, bias]]
        ctx.save_for_backward(qx, qw)
        ctx.dtype = dtype
        return rounded(qx @ qw.T + qb, dtype)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        g = rounded(grad, ctx.dtype).double()
        x2, g2 = x.reshape(-1, x.shape[-1]), g.reshape(-1, g.shape[-1])
        return (
            rounded(g @ w, ctx.dtype),
            rounded(g2.T @ x2, ctx.dtype),
            rounded(g2.sum(0), ctx.dtype),
            None,
        )


class OracleTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype):
        y = rounded(x.double().tanh(), dtype)
        ctx.save_for_backward(y)
        ctx.dtype = dtype
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return rounded(grad.double() * (1 - y.double().square()), ctx.dtype), None


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        torch.nn.Linear(CONFIG["width"], CONFIG["hidden"]),
                        torch.nn.Linear(CONFIG["hidden"], CONFIG["width"]),
                    ]
                )
                for _ in range(CONFIG["blocks"])
            ]
        )

    def forward(self, x, oracle_dtype=None, trace=None):
        for up, down in self.layers:
            if oracle_dtype is None:
                h = up(x)
                z = down(h.tanh())
                if trace is not None:
                    trace.extend([str(h.dtype), str(z.dtype)])
            else:
                h = OracleLinear.apply(x, up.weight, up.bias, oracle_dtype)
                z = OracleLinear.apply(
                    OracleTanh.apply(h, oracle_dtype),
                    down.weight,
                    down.bias,
                    oracle_dtype,
                )
            x = x + z * 0.125
        return x


def tree_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual.cpu(), expected.cpu(), atol=0, rtol=0, equal_nan=True
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected:
            tree_exact(actual[k], expected[k])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            tree_exact(a, e)
    else:
        assert actual == expected, (actual, expected)


def tensor_state_hash(model, opt, scaler):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
        for k, value in sorted(opt.state[p].items()):
            h.update(k.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    h.update(json.dumps(scaler.state_dict(), sort_keys=True).encode())
    return h.hexdigest()


def validate_scaler(scaler):
    """Refuse divergent public scaler states collectively before any scaled backward."""
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, scaler.state_dict())
    for state in states:
        if state and (not math.isfinite(state["scale"]) or state["scale"] <= 0):
            raise ValueError("Invalid AMP scale")
    if any(s != states[0] for s in states):
        raise ValueError("Scaler states differ between ranks")


def norm_check(actual, wanted, limit):
    a, w = actual.detach().cpu().double(), wanted.detach().cpu().double()
    assert torch.isfinite(a).all() and torch.isfinite(w).all()
    relative = (a - w).norm().item() / max(w.norm().item(), 1e-12)
    assert relative <= limit, (relative, limit, a.shape)
    return relative


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", choices=["bf16", "fp16"], required=True)
    p.add_argument("--mode", choices=["sync", "no-sync"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    CONFIG.update(dtype=a.dtype, mode=a.mode)
    result = dict(status="running", config=dict(CONFIG), checks=[], metrics=[])
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
        dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
        limit = 0.02 if a.dtype == "bf16" else 0.004
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

        def comm_hook(state, bucket):
            state["calls"] += 1
            state["bytes"] += bucket.buffer().numel() * bucket.buffer().element_size()
            assert bucket.buffer().dtype == torch.float32
            return allreduce_hook(None, bucket)

        step_calls = [0]

        def make():
            model = DDP(
                copy.deepcopy(template).cuda(),
                device_ids=[0],
                forward_sync_buffers=False,
                gradient_as_bucket_view=True,
            )
            model.register_comm_hook(traffic, comm_hook)
            opt = torch.optim.AdamW(
                model.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
            )

            def record_step(*_):
                step_calls[0] += 1

            opt.register_step_post_hook(record_step)
            scaler = torch.amp.GradScaler(
                "cuda", enabled=a.dtype == "fp16", init_scale=16, growth_interval=2
            )
            return model, opt, scaler

        model, opt, scaler = make()
        result.update(
            rank=rank,
            world=world,
            torch=torch.__version__,
            script_sha256=sha(Path(__file__)),
            reused_d05_sha256=sha(Path(__file__).with_name("distributed_d05_train.py")),
            parameter_elements=elements,
            gradient_dtype="torch.float32",
            cpu_affinity=sorted(os.sched_getaffinity(0)),
        )
        shape = (CONFIG["global_batch"], CONFIG["tokens"], CONFIG["width"])
        indices = torch.arange(math.prod(shape)).reshape(shape)
        offsets = torch.arange(shape[0]).reshape(-1, 1, 1).float() / 97
        data = (indices % 31 - 15).float() / 32 + offsets
        targets = (indices % 29 - 14).float() / 32 - offsets / 2
        n = shape[0] // world
        x = data[rank * n : (rank + 1) * n].cuda().requires_grad_()
        target = targets[rank * n : (rank + 1) * n].cuda()
        windows = []

        def begin(index, fault=None):
            validate_scaler(scaler)
            opt.zero_grad(set_to_none=True)
            x.grad = None
            total = torch.zeros((), device="cuda")
            micro_records = []
            traces = []
            for i in range(n):
                before = dict(traffic)
                # Inject only before the final synchronized backward: RCCL propagates the bad gradient.
                if fault is not None and rank == 1 and i == n - 1:
                    next(model.parameters()).grad.reshape(-1)[0] = float(fault)
                context = (
                    model.no_sync()
                    if a.mode == "no-sync" and i + 1 < n
                    else nullcontext()
                )
                with context:
                    with torch.autocast("cuda", dtype=dtype):
                        out = model(x[i : i + 1], trace=traces)
                        loss = (
                            out.float() - target[i : i + 1] * (1 + index / 256)
                        ).square().mean() / n
                    assert loss.dtype == torch.float32 and out.dtype == torch.float32
                    scaler.scale(loss).backward()
                total += loss.detach()
                delta = {k: traffic[k] - before[k] for k in traffic}
                active = a.mode == "sync" or i + 1 == n
                assert delta["bytes"] == (elements * 4 if active else 0)
                assert bool(delta["calls"]) == active
                micro_records.append(delta)
            assert traces and set(traces) == {str(dtype)}
            scaler.unscale_(opt)
            flags = [None] * world
            finite = all(
                torch.isfinite(p.grad).all().item() for p in model.parameters()
            )
            dist.all_gather_object(flags, finite)
            assert all(
                v == flags[0] for v in flags
            ), "Post-reduction rank-local gradient divergence"
            assert finite == (fault is None)
            windows.append(
                dict(
                    index=index,
                    fault=fault,
                    finite=finite,
                    scale_before=scaler.get_scale(),
                    micros=micro_records,
                )
            )
            return total, finite

        def finish(finite):
            before = step_calls[0]
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            elif finite:
                opt.step()
            assert step_calls[0] - before == int(finite)
            windows[-1]["scale_after"] = scaler.get_scale()
            states = [None] * world
            dist.all_gather_object(states, tensor_state_hash(model, opt, scaler))
            assert states[0] == states[1], "Parameter/AdamW/scaler replica divergence"
            return scaler.state_dict()

        for index in range(2):
            oracle.load_state_dict(
                {k: v.detach().cpu() for k, v in model.module.state_dict().items()}
            )
            oracle.zero_grad(set_to_none=True)
            refopt.zero_grad(set_to_none=True)
            scale = scaler.get_scale()
            expected_loss = torch.zeros(())
            cpu_x = data.clone().requires_grad_()
            for i in range(shape[0]):
                prediction = oracle(cpu_x[i : i + 1], oracle_dtype=dtype)
                cpu_loss = (
                    prediction - targets[i : i + 1] * (1 + index / 256)
                ).square().mean() / n
                (cpu_loss * scale).backward()
                expected_loss += cpu_loss.detach() / world
            loss, finite = begin(index)
            loss_global = loss.detach().clone()
            dist.all_reduce(loss_global)
            loss_global /= world
            torch.testing.assert_close(
                loss_global.cpu(),
                expected_loss,
                atol=2e-5,
                rtol=0.002 if a.dtype == "bf16" else 0.0005,
            )
            errors = []
            for actual, wanted, cpu_param in zip(
                model.parameters(), oracle.parameters(), opt_oracle.parameters()
            ):
                errors.append(
                    norm_check(actual.grad, wanted.grad / (world * scale), limit)
                )
                # Independent CPU FP64 AdamW verifies the optimizer on the actual admitted AMP gradient.
                cpu_param.grad = actual.grad.detach().cpu().double().clone()
            input_error = norm_check(
                x.grad / scale, cpu_x.grad[rank * n : (rank + 1) * n] / scale, limit
            )
            finish(finite)
            refopt.step()
            for actual, wanted in zip(model.parameters(), opt_oracle.parameters()):
                torch.testing.assert_close(
                    actual.detach().cpu().double(), wanted, atol=4e-6, rtol=5e-5
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
                tree_exact(opt.state[actual]["step"], refopt.state[wanted]["step"])
            result["metrics"].append(
                dict(
                    index=index,
                    maximum_parameter_gradient_relative_L2=max(errors),
                    input_gradient_relative_L2=input_error,
                )
            )
            result["checks"].append(
                f"CPU-quantized-FP64-loss-input-and-all-param-gradients-{index}"
            )
            result["checks"].append(f"CPU-FP64-AdamW-actual-admitted-gradient-{index}")
        if scaler.is_enabled():
            assert (
                scaler.state_dict()["scale"] == 32
                and scaler.state_dict()["_growth_tracker"] == 0
            )
        result["checks"].append("two-finite-updates-and-scaler-growth")

        # Collectively reject scaler mismatches BEFORE the next scaled forward/backward.
        if scaler.is_enabled():
            saved = scaler.state_dict()
            for field in ["scale", "_growth_tracker"]:
                wrong = dict(saved)
                wrong[field] = (
                    wrong[field] * 2 if field == "scale" else wrong[field] + 1
                )
                if rank == 1:
                    scaler.load_state_dict(wrong)
                try:
                    validate_scaler(scaler)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Divergent scaler state accepted")
                scaler.load_state_dict(saved)
            result["checks"].append(
                "two-collective-prebackward-scaler-mismatch-refusals"
            )

        for index, fault in [(2, "inf"), (3, "nan")]:
            saved_params = [p.detach().cpu().clone() for p in model.parameters()]
            saved_opt = copy.deepcopy(opt.state_dict())
            previous_scale = scaler.get_scale()
            _, finite = begin(index, fault)
            finish(finite)
            for actual, wanted in zip(model.parameters(), saved_params):
                tree_exact(actual, wanted)
            tree_exact(opt.state_dict(), saved_opt)
            if scaler.is_enabled():
                assert (
                    scaler.get_scale() == previous_scale / 2
                    and scaler.state_dict()["_growth_tracker"] == 0
                )
            result["checks"].append(f"rank1-{fault}-propagation-both-skip-exact-state")
        loss, finite = begin(4)
        finish(finite)
        result["checks"].append("finite-recovery-after-nonfinite-skips")

        folder = a.output.parent / "checkpoint"
        folder.mkdir()
        part = folder / f"rank-{rank}.pt"
        payload = dict(
            params=[p.detach().cpu().clone() for p in model.parameters()],
            optimizer=opt.state_dict(),
            scaler=scaler.state_dict(),
            rank=rank,
            world=world,
            config=dict(CONFIG),
            next_index=5,
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
        expected_loss, finite = begin(5)
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
            and loaded["next_index"] == 5
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
        result["checks"].append(
            "actual-autocast-dtypes-and-FP32-gradient-reduction-bytes"
        )
        result["windows"] = windows
        result["memory"] = dict(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        result["worker_status"] = Path("/proc/self/status").read_text()
        result["host_meminfo"] = Path("/proc/meminfo").read_text()
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
