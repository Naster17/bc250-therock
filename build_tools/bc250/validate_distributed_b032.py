#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Two ranks sharing one gfx1013: Gloo host-staged GPU tensors and DDP references."""

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
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
    report = dict(status="running", rank=rank, pid=os.getpid(), checks=[])
    path = directory / f"rank{rank}.json"

    def passed(name):
        report["checks"].append(name)
        path.write_text(json.dumps(report, indent=2) + "\n")

    def exact(actual, expected):
        assert actual.device.type == "cuda"
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    try:
        dist.init_process_group("gloo", timeout=timedelta(seconds=45))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
                torch.int32,
                torch.int64,
                torch.complex64,
                torch.complex128,
            ):
                for count in (1, 17, 1025):
                    # Small exactly represented values avoid low-precision oracle rounding.
                    reference = (torch.arange(count) % 7).to(dtype)
                    if dtype.is_complex:
                        reference += 0.25j
                    storage = torch.full((count + 10,), -19, device="cuda", dtype=dtype)
                    x = storage[5:-5]
                    x.copy_(reference + rank)
                    work = dist.all_reduce(x, async_op=True)
                    assert work.wait(timeout=timedelta(seconds=30))
                    # Consume on the submitting stream before any CPU synchronization.
                    consumed = x * 2
                    exact(consumed, (2 * reference + 1) * 2)
                    exact(storage[:5], torch.full((5,), -19, dtype=dtype))
                    exact(storage[-5:], torch.full((5,), -19, dtype=dtype))
                    passed(f"sum-offset-guards-private-stream-{dtype}-{count}")
            for name, op, expected in (
                ("product", dist.ReduceOp.PRODUCT, 6),
                ("min", dist.ReduceOp.MIN, 2),
                ("max", dist.ReduceOp.MAX, 3),
                ("avg", dist.ReduceOp.AVG, 2.5),
            ):
                x = torch.full((19,), rank + 2, device="cuda", dtype=torch.float64)
                dist.all_reduce(x, op=op)
                exact(x, torch.full((19,), expected, dtype=torch.float64))
                passed("allreduce-" + name)
            x = torch.full((7,), rank, device="cuda", dtype=torch.int64)
            dist.broadcast(x, src=1)
            exact(x, torch.ones(7, dtype=torch.int64))
            passed("broadcast")
            x = torch.arange(6, device="cuda", dtype=torch.float64) + rank * 10
            outputs = [torch.empty_like(x) for _ in range(2)]
            dist.all_gather(outputs, x)
            for r, y in enumerate(outputs):
                exact(y, torch.arange(6, dtype=torch.float64) + r * 10)
            passed("allgather")
            outputs = [torch.empty_like(x) for _ in range(2)] if rank == 0 else None
            dist.gather(x, gather_list=outputs, dst=0)
            if rank == 0:
                for r, y in enumerate(outputs):
                    exact(y, torch.arange(6, dtype=torch.float64) + r * 10)
            passed("gather")
            y = torch.empty_like(x)
            dist.scatter(y, scatter_list=[x, x + 10] if rank == 0 else None, src=0)
            exact(y, torch.arange(6, dtype=torch.float64) + rank * 10)
            passed("scatter")
            dist.reduce(x, dst=0)
            if rank == 0:
                exact(x, 2 * torch.arange(6, dtype=torch.float64) + 10)
            passed("reduce")
            x = torch.arange(6, device="cuda", dtype=torch.float64) + rank * 10
            y = torch.empty(3, device="cuda", dtype=torch.float64)
            dist.reduce_scatter_tensor(y, x)
            exact(
                y, 2 * torch.arange(rank * 3, (rank + 1) * 3, dtype=torch.float64) + 10
            )
            passed("reduce-scatter-tensor")
            y = torch.empty_like(x)
            dist.all_to_all_single(y, x)
            exact(
                y,
                torch.cat(
                    [
                        torch.arange(rank * 3, (rank + 1) * 3, dtype=torch.float64)
                        + 10 * r
                        for r in range(2)
                    ]
                ),
            )
            passed("all-to-all-single")
            from torch.distributed.nn.functional import all_reduce

            for dtype in (torch.float32, torch.float64):
                x = torch.tensor(
                    [rank + 1.0, rank + 2.0],
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                y = all_reduce(x)
                exact(y, torch.tensor([3.0, 5.0], dtype=dtype))
                (y * (rank + 1)).sum().backward()
                exact(x.grad, torch.full((2,), 3, dtype=dtype))
                passed("functional-first-gradient-" + str(dtype))
            for dtype in (torch.float32, torch.float64):
                for optimizer_name in ("sgd", "adamw"):

                    def model():
                        return torch.nn.Sequential(
                            torch.nn.Linear(3, 5),
                            torch.nn.Tanh(),
                            torch.nn.Linear(5, 2),
                        ).to(dtype)

                    torch.manual_seed(1032)
                    reference = model()
                    torch.manual_seed(1032 + rank)
                    native = model().cuda()
                    ddp = torch.nn.parallel.DistributedDataParallel(
                        native, device_ids=[0]
                    )
                    rtol, atol = (
                        (3e-5, 3e-7) if dtype == torch.float32 else (2e-11, 2e-13)
                    )

                    def compare(a, b):
                        assert a.device.type == "cuda" and b.device.type == "cpu"
                        torch.testing.assert_close(a.cpu(), b, rtol=rtol, atol=atol)

                    for a, b in zip(native.parameters(), reference.parameters()):
                        compare(a, b)
                    if optimizer_name == "sgd":
                        make = lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.8)
                    else:
                        make = lambda p: torch.optim.AdamW(
                            p, lr=0.002, weight_decay=0.01
                        )
                    opt, refopt = make(native.parameters()), make(
                        reference.parameters()
                    )
                    for step in range(2):
                        data = (
                            torch.linspace(-0.5, 0.8, 24, dtype=dtype).reshape(8, 3)
                            + step * 0.1
                        )
                        targets = torch.linspace(0.4, -0.2, 16, dtype=dtype).reshape(
                            8, 2
                        )
                        actual = ddp(data[rank * 4 : (rank + 1) * 4].cuda())
                        expected = reference(data)
                        compare(actual, expected[rank * 4 : (rank + 1) * 4])
                        opt.zero_grad()
                        refopt.zero_grad()
                        (
                            actual - targets[rank * 4 : (rank + 1) * 4].cuda()
                        ).square().mean().backward()
                        (expected - targets).square().mean().backward()
                        for a, b in zip(native.parameters(), reference.parameters()):
                            compare(a.grad, b.grad)
                        opt.step()
                        refopt.step()
                        for a, b in zip(native.parameters(), reference.parameters()):
                            compare(a, b)
                    passed(f"gpu-ddp-two-updates-{dtype}-{optimizer_name}")
                    del ddp, native, reference, opt, refopt
        torch.cuda.synchronize()
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
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.directory)
        return
    report = dict(status="running", checks=[], workers=[])
    try:
        directory = args.output.parent / "owned-gpu-ranks"
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
            GLOO_DEVICE_TRANSPORT="TCP",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        with (args.output.parent / "gpu-torchrun.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=240
            )
        assert result.returncode == 0, result.returncode
        workers = [
            json.loads((directory / f"rank{r}.json").read_text()) for r in range(2)
        ]
        report["workers"] = workers
        assert all(w["status"] == "passed" for w in workers)
        assert (
            workers[0]["pid"] != workers[1]["pid"]
            and workers[0]["checks"] == workers[1]["checks"]
        )
        report["checks"] = workers[0]["checks"]
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
