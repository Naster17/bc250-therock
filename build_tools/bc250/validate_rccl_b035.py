#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Isolated Torch RCCL rank-one integration; no multi-device transport claim."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(output):
    import torch
    import torch.distributed as dist

    report = dict(status="running", checks=[], world_size=1)

    def passed(name):
        report["checks"].append(name)
        output.write_text(json.dumps(report, indent=2) + "\n")

    def exact(actual, reference):
        assert actual.device.type == "cuda" and reference.device.type == "cpu"
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        assert torch.cuda.device_count() == 1
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
        assert dist.is_nccl_available() and dist.is_gloo_available()
        report["torch"] = torch.__version__
        report["rccl_version"] = torch.cuda.nccl.version()
        assert report["rccl_version"] == (2, 32, 3)
        passed("native-device-and-backend-version")
        store = output.parent / "rank-one-store"
        dist.init_process_group(
            "nccl",
            init_method="file://" + str(store),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=45),
            device_id=torch.device("cuda:0"),
        )
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
                torch.int8,
                torch.uint8,
                torch.int32,
                torch.int64,
            ):
                for count in (1, 17, 1025):
                    ref = (torch.arange(count) % 7).to(dtype)
                    storage = torch.full((count + 10,), 19, device="cuda", dtype=dtype)
                    x = storage[5:-5]
                    x.copy_(ref)
                    work = dist.all_reduce(x, async_op=True)
                    assert work.wait(timeout=timedelta(seconds=30))
                    exact(x * 2, ref * 2)
                    exact(storage[:5], torch.full((5,), 19, dtype=dtype))
                    exact(storage[-5:], torch.full((5,), 19, dtype=dtype))
                    passed(f"sum-offset-guards-stream-{dtype}-{count}")
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for count in (1, 17, 1025):
                    for residence in ("host", "device"):
                        ref = (((torch.arange(count) % 7).float() + 1) / 4).to(dtype)
                        x = ref.cuda()
                        scalar = (
                            2.0
                            if residence == "host"
                            else torch.tensor([2], device="cuda", dtype=dtype)
                        )
                        op = dist._make_nccl_premul_sum(scalar)
                        work = dist.all_reduce(x, op=op, async_op=True)
                        assert work.wait(timeout=timedelta(seconds=30))
                        exact(x * 2, ref * 4)
                        passed(f"premul-{dtype}-{count}-{residence}")
            for name, op in (
                ("prod", dist.ReduceOp.PRODUCT),
                ("min", dist.ReduceOp.MIN),
                ("max", dist.ReduceOp.MAX),
                ("avg", dist.ReduceOp.AVG),
            ):
                ref = torch.arange(17, dtype=torch.float64) / 4
                x = ref.cuda()
                dist.all_reduce(x, op=op)
                exact(x, ref)
                passed("identity-" + name)
            ref = torch.arange(17, dtype=torch.float32) / 4
            x = ref.cuda()
            dist.broadcast(x, src=0)
            exact(x, ref)
            passed("identity-broadcast")
            dist.reduce(x, dst=0)
            exact(x, ref)
            passed("identity-reduce")
            y = torch.empty_like(x)
            dist.all_gather([y], x)
            exact(y, ref)
            passed("identity-allgather")
            dist.all_gather_into_tensor(y, x)
            exact(y, ref)
            passed("identity-allgather-tensor")
            dist.reduce_scatter_tensor(y, x)
            exact(y, ref)
            passed("identity-reduce-scatter-tensor")
            dist.all_to_all_single(y, x)
            exact(y, ref)
            passed("identity-alltoall-single")
            from torch.distributed.nn.functional import all_reduce

            for dtype in (torch.float32, torch.float64):
                ref = torch.arange(17, dtype=dtype) / 8
                x = ref.cuda().requires_grad_()
                y = all_reduce(x)
                weights = torch.linspace(-0.5, 0.5, 17, dtype=dtype)
                (y.square() * weights.cuda()).sum().backward()
                exact(y, ref)
                exact(x.grad, 2 * ref * weights)
                passed(f"functional-first-gradient-{dtype}")
            for dtype in (torch.float32, torch.float64):
                reference = torch.nn.Linear(3, 2, dtype=dtype)
                with torch.no_grad():
                    reference.weight.copy_(
                        torch.arange(6, dtype=dtype).reshape(2, 3) / 16
                    )
                    reference.bias.copy_(torch.tensor([0.25, -0.25], dtype=dtype))
                native = torch.nn.Linear(3, 2, dtype=dtype).cuda()
                native.load_state_dict(reference.state_dict())
                ddp = torch.nn.parallel.DistributedDataParallel(native, device_ids=[0])
                opt = torch.optim.SGD(native.parameters(), lr=0.02, momentum=0.8)
                refopt = torch.optim.SGD(reference.parameters(), lr=0.02, momentum=0.8)
                tol = (
                    dict(rtol=3e-5, atol=3e-7)
                    if dtype == torch.float32
                    else dict(rtol=2e-11, atol=2e-13)
                )
                for step in range(2):
                    data = (
                        torch.linspace(-0.5, 0.8, 12, dtype=dtype).reshape(4, 3)
                        + step / 8
                    )
                    targets = torch.linspace(0.4, -0.2, 8, dtype=dtype).reshape(4, 2)
                    actual, expected = ddp(data.cuda()), reference(data)
                    torch.testing.assert_close(actual.cpu(), expected, **tol)
                    opt.zero_grad()
                    refopt.zero_grad()
                    (actual - targets.cuda()).square().mean().backward()
                    (expected - targets).square().mean().backward()
                    for a, b in zip(native.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad.cpu(), b.grad, **tol)
                    opt.step()
                    refopt.step()
                    for a, b in zip(native.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.cpu(), b, **tol)
                passed(f"ddp-two-sgd-updates-{dtype}")
                del ddp, native, reference, opt, refopt
        torch.cuda.synchronize()
        dist.barrier()
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/librccl.so" in line
            }
        )
        expected = Path(
            "/home/nik/bc250work/rocm-10.2-gfx1013/rccl-b034/install-r1/lib/librccl.so"
        ).resolve()
        assert libraries and all(
            Path(p).resolve() == expected for p in libraries
        ), libraries
        report["loaded_libraries"] = libraries
        passed("loaded-isolated-dso-and-lifecycle")
        assert len(report["checks"]) == 64
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
        return
    env = dict(os.environ)
    env.update(
        LD_LIBRARY_PATH="/home/nik/bc250work/rocm-10.2-gfx1013/rccl-b034/install-r1/lib:"
        + env.get("LD_LIBRARY_PATH", ""),
        NCCL_SOCKET_IFNAME="lo",
        NCCL_IB_DISABLE="1",
        NCCL_P2P_DISABLE="1",
        NCCL_CUMEM_ENABLE="0",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_DEBUG="INFO",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
    )
    subprocess.run(
        [sys.executable, __file__, "--worker", "--output", str(args.output)],
        env=env,
        check=True,
        timeout=180,
    )


if __name__ == "__main__":
    main()
