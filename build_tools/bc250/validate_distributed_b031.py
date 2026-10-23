#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Two local CPU Gloo ranks: exact collectives, first gradients and DDP updates."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.distributed as dist


def worker(directory):
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2
    report = dict(status="running", rank=rank, pid=os.getpid(), checks=[])
    path = directory / f"rank{rank}.json"

    def passed(name):
        report["checks"].append(name)
        path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        dist.init_process_group("gloo", timeout=timedelta(seconds=45))
        assert dist.get_backend() == "gloo" and dist.get_world_size() == 2
        passed("init-gloo")
        for dtype in (
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.complex64,
            torch.complex128,
        ):
            base = torch.arange(11).to(dtype)
            x = base + rank
            if dtype.is_complex:
                x += (rank + 1) * 0.25j
            work = dist.all_reduce(x, async_op=True)
            assert work.wait(timeout=timedelta(seconds=30))
            expected = 2 * base + 1
            if dtype.is_complex:
                expected += 0.75j
            torch.testing.assert_close(x, expected, rtol=0, atol=0)
            assert x.device.type == "cpu" and x.dtype == dtype
            passed("allreduce-sum-" + str(dtype))
        for name, op, expected in (
            ("product", dist.ReduceOp.PRODUCT, 6),
            ("min", dist.ReduceOp.MIN, 2),
            ("max", dist.ReduceOp.MAX, 3),
        ):
            x = torch.full((7,), rank + 2, dtype=torch.float64)
            dist.all_reduce(x, op=op)
            torch.testing.assert_close(x, torch.full_like(x, expected), rtol=0, atol=0)
            passed("allreduce-" + name)
        x = torch.full((5,), rank, dtype=torch.int64)
        dist.broadcast(x, src=1)
        assert torch.equal(x, torch.ones_like(x))
        passed("broadcast")
        gathered = [torch.empty(3, dtype=torch.float32) for _ in range(world)]
        dist.all_gather(gathered, torch.arange(3, dtype=torch.float32) + rank * 10)
        for source, t in enumerate(gathered):
            assert torch.equal(t, torch.arange(3, dtype=torch.float32) + source * 10)
        passed("allgather")
        gathered = (
            [torch.empty(3, dtype=torch.int64) for _ in range(world)]
            if rank == 0
            else None
        )
        dist.gather(
            torch.arange(3, dtype=torch.int64) + rank * 10, gather_list=gathered, dst=0
        )
        if rank == 0:
            for source, t in enumerate(gathered):
                assert torch.equal(t, torch.arange(3, dtype=torch.int64) + source * 10)
        passed("gather")
        target = torch.empty(3, dtype=torch.int64)
        dist.scatter(
            target,
            scatter_list=(
                [torch.full((3,), r + 5, dtype=torch.int64) for r in range(world)]
                if rank == 1
                else None
            ),
            src=1,
        )
        assert torch.equal(target, torch.full_like(target, rank + 5))
        passed("scatter")
        x = torch.ones(5, dtype=torch.float64) * (rank + 1)
        dist.reduce(x, dst=0)
        if rank == 0:
            assert torch.equal(x, torch.full_like(x, 3))
        passed("reduce")
        x = torch.arange(6, dtype=torch.float64) + 10 * rank
        y = torch.empty(3, dtype=torch.float64)
        dist.reduce_scatter_tensor(y, x)
        assert torch.equal(
            y, 2 * torch.arange(rank * 3, (rank + 1) * 3, dtype=torch.float64) + 10
        )
        passed("reduce-scatter-tensor")
        y = torch.empty_like(x)
        dist.all_to_all_single(y, x)
        expected = torch.cat(
            [
                torch.arange(rank * 3, (rank + 1) * 3, dtype=torch.float64) + 10 * r
                for r in range(world)
            ]
        )
        assert torch.equal(y, expected)
        passed("all-to-all-single")
        if rank == 0:
            dist.send(torch.tensor([2, 4, 6]), dst=1, tag=7)
            y = torch.empty(3, dtype=torch.int64)
            dist.recv(y, src=1, tag=8)
            assert torch.equal(y, torch.tensor([3, 5, 7]))
        else:
            y = torch.empty(3, dtype=torch.int64)
            dist.recv(y, src=0, tag=7)
            assert torch.equal(y, torch.tensor([2, 4, 6]))
            dist.send(y + 1, dst=0, tag=8)
        passed("point-to-point")
        subgroup = dist.new_group(
            ranks=[0, 1], backend="gloo", timeout=timedelta(seconds=30)
        )
        x = torch.tensor([rank + 1.0])
        dist.all_reduce(x, group=subgroup)
        assert x.item() == 3
        dist.destroy_process_group(subgroup)
        dist.monitored_barrier(timeout=timedelta(seconds=30))
        passed("subgroup-barrier")
        from torch.distributed.nn.functional import all_reduce

        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([rank + 1.0, rank + 2.0], dtype=dtype, requires_grad=True)
            y = all_reduce(x)
            expected = torch.tensor([3.0, 5.0], dtype=dtype)
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            (y * (rank + 1)).sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, 3), rtol=0, atol=0)
            passed("functional-first-gradient-" + str(dtype))
        # Explicit non-distributed global-batch reference, two optimizer steps.
        for dtype in (torch.float32, torch.float64):
            for optimizer_name in ("sgd", "adamw"):
                torch.manual_seed(1031 + rank)
                model = torch.nn.Sequential(
                    torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)
                ).to(dtype)
                ddp = torch.nn.parallel.DistributedDataParallel(model)
                torch.manual_seed(1031)
                reference = torch.nn.Sequential(
                    torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)
                ).to(dtype)
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if optimizer_name == "sgd":
                    make_optimizer = lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.8)
                else:
                    make_optimizer = lambda p: torch.optim.AdamW(
                        p, lr=0.002, weight_decay=0.01
                    )
                optimizer = make_optimizer(model.parameters())
                ref_optimizer = make_optimizer(reference.parameters())
                for step in range(2):
                    data = (
                        torch.linspace(-0.5, 0.8, 24, dtype=dtype).reshape(8, 3)
                        + step * 0.1
                    )
                    targets = torch.linspace(0.4, -0.2, 16, dtype=dtype).reshape(8, 2)
                    local_data = data[rank * 4 : (rank + 1) * 4]
                    actual = ddp(local_data)
                    expected = reference(data)
                    rtol, atol = (
                        (3e-5, 3e-7) if dtype == torch.float32 else (2e-11, 2e-13)
                    )
                    torch.testing.assert_close(
                        actual,
                        expected[rank * 4 : (rank + 1) * 4],
                        rtol=rtol,
                        atol=atol,
                    )
                    loss = (actual - targets[rank * 4 : (rank + 1) * 4]).square().mean()
                    ref_loss = (expected - targets).square().mean()
                    optimizer.zero_grad()
                    ref_optimizer.zero_grad()
                    loss.backward()
                    ref_loss.backward()
                    for a, b in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=rtol, atol=atol)
                    optimizer.step()
                    ref_optimizer.step()
                    for a, b in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
                passed(f"ddp-two-updates-{dtype}-{optimizer_name}")
                del ddp, model, reference, optimizer, ref_optimizer
        dist.barrier()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--nccl-available", action="store_true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.directory)
        return
    assert args.output is not None
    torch.set_num_threads(1)
    report = dict(status="running", checks=[], workers=[])
    try:
        assert dist.is_available() and dist.is_gloo_available()
        assert dist.is_nccl_available() == args.nccl_available
        assert not dist.is_mpi_available()
        report["checks"].append("build-feature-contract")
        for name, store in (
            ("hash", dist.HashStore()),
            ("file", dist.FileStore(str(args.output.parent / "owned-file-store"), 1)),
            (
                "tcp-libuv",
                dist.TCPStore(
                    "127.0.0.1",
                    0,
                    1,
                    True,
                    timedelta(seconds=10),
                    wait_for_workers=False,
                    use_libuv=True,
                ),
            ),
        ):
            store.set("key", "value")
            assert store.get("key") == b"value"
            assert store.add("number", 2) == 2 and store.add("number", 3) == 5
            assert store.compare_set("key", "value", "next") == b"next"
            assert store.get("key") == b"next"
            if name == "tcp-libuv":
                assert store.libuvBackend and store.port > 0
            report["checks"].append("store-" + name)
        directory = args.output.parent / "owned-ranks"
        directory.mkdir()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--max-restarts=0",
            str(Path(__file__).resolve()),
            "--worker",
            "--directory",
            str(directory),
        ]
        report["command"] = command
        env = dict(
            os.environ,
            GLOO_SOCKET_IFNAME="lo",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        with (args.output.parent / "torchrun.log").open("w") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=180
            )
        assert result.returncode == 0, result.returncode
        workers = [
            json.loads((directory / f"rank{r}.json").read_text()) for r in range(2)
        ]
        assert all(w["status"] == "passed" for w in workers)
        assert (
            workers[0]["pid"] != workers[1]["pid"]
            and workers[0]["checks"] == workers[1]["checks"]
        )
        report["workers"] = workers
        report["checks"] += ["two-ranks-" + name for name in workers[0]["checks"]]
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
