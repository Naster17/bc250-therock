#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""One-device public RCCL contracts; identities do not qualify transport."""

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
    from torch.cuda import nccl

    report = dict(status="running", checks=[], refusals={}, world_size=1)

    def passed(name):
        report["checks"].append(name)
        output.write_text(json.dumps(report, indent=2) + "\n")

    def exact(actual, expected):
        assert actual.device.type == "cuda" and expected.device.type == "cpu"
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def refuse(name, operation, text):
        try:
            operation()
        except (RuntimeError, ValueError, TypeError) as error:
            assert text.lower() in str(error).lower(), (name, str(error))
            report["refusals"][name] = str(error)
        else:
            raise AssertionError("Expected refusal: " + name)
        passed(name)

    def lowlevel(op, x, y, stream):
        if op == "allreduce":
            nccl.all_reduce([x], [y], streams=[stream])
        elif op == "reduce":
            nccl.reduce([x], output=y, streams=[stream])
        elif op == "gather":
            nccl.all_gather([x], [y], streams=[stream])
        else:
            nccl.reduce_scatter([x], [y], streams=[stream])

    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        assert torch.cuda.device_count() == 1
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
        assert dist.is_nccl_available() and nccl.version() == (2, 32, 3)
        stream = torch.cuda.Stream()
        types = (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.int64,
        )
        with torch.cuda.stream(stream):
            for dtype in types:
                for count in (17, 257):
                    for alias in (False, True):
                        for op in ("allreduce", "reduce", "gather", "scatter"):
                            ref = (torch.arange(count) % 7).to(dtype)
                            storage = torch.full(
                                (count + 10,), 19, device="cuda", dtype=dtype
                            )
                            x = storage[5:-5]
                            x.copy_(ref)
                            target = storage if alias else torch.full_like(storage, 23)
                            y = target[5:-5]
                            lowlevel(op, x, y, stream)
                            consumed = y * 2
                            exact(consumed, ref * 2)
                            exact(x, ref)
                            for buffer, marker in (
                                (storage, 19),
                                (target, 19 if alias else 23),
                            ):
                                exact(buffer[:5], torch.full((5,), marker, dtype=dtype))
                                exact(
                                    buffer[-5:], torch.full((5,), marker, dtype=dtype)
                                )
                            passed(f"identity-{op}-{dtype}-{count}-alias{alias}")
                x = torch.arange(17).to(dtype).cuda()
                nccl.broadcast([x], streams=[stream])
                exact(x, torch.arange(17).to(dtype))
                passed(f"identity-broadcast-{dtype}")
            for op in ("allreduce", "reduce", "gather", "scatter", "broadcast"):
                x = torch.empty(0, device="cuda")
                y = torch.empty_like(x)
                if op == "broadcast":
                    nccl.broadcast([x], streams=[stream])
                else:
                    lowlevel(op, x, y, stream)
                exact(y, torch.empty(0))
                passed("empty-" + op)
            x = torch.arange(17, device="cuda", dtype=torch.float32)
            y = torch.empty_like(x)
            for name, tensors, expected in (
                ("one-device", [x], True),
                ("cpu", [x.cpu()], False),
                ("strided", [x[::2]], False),
                ("duplicate-device", [x, x], False),
            ):
                assert nccl.is_available(tensors) == expected
                passed("availability-" + name)
            for name, action, text in (
                ("tensor-not-sequence", lambda: nccl.all_reduce(x), "collection"),
                ("empty-sequence", lambda: nccl.all_reduce([]), "empty"),
                ("list-length", lambda: nccl.all_reduce([x], []), "length"),
                ("cpu-input", lambda: nccl.all_reduce([x.cpu()]), "cuda dense"),
                ("strided-input", lambda: nccl.all_reduce([x[::2]]), "contiguous"),
                ("dtype-mismatch", lambda: nccl.all_reduce([x], [y.double()]), "dtype"),
                (
                    "size-mismatch",
                    lambda: nccl.all_reduce([x], [torch.empty(18, device="cuda")]),
                    "size",
                ),
                (
                    "duplicate-input-device",
                    lambda: nccl.all_reduce([x, x], [y, y]),
                    "unique devices",
                ),
                ("stream-length", lambda: nccl.all_reduce([x], streams=[]), "streams"),
                ("root-range", lambda: nccl.broadcast([x], root=1), "invalid root"),
                (
                    "unsupported-int16",
                    lambda: nccl.all_reduce([x.to(torch.int16)]),
                    "Unconvertible",
                ),
                (
                    "unsupported-complex",
                    lambda: nccl.all_reduce([x.to(torch.complex64)]),
                    "Unconvertible",
                ),
            ):
                refuse("lowlevel-" + name, action, text)
        for cycle in range(3):
            store = output.parent / f"owned-store-{cycle}"
            dist.init_process_group(
                "nccl",
                init_method="file://" + str(store),
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=45),
                device_id=torch.device("cuda:0"),
            )
            assert dist.get_backend() == "nccl" and dist.get_world_size() == 1
            with torch.cuda.stream(stream):
                ref = torch.arange(17, dtype=torch.float32) / 4 + 0.25
                x = ref.cuda()
                dist.all_reduce(
                    x, op=dist._make_nccl_premul_sum(2.0), async_op=True
                ).wait()
                exact(x, ref * 2)
                passed(f"reinit-premul-{cycle}")
                if cycle == 0:
                    group = dist.new_group(
                        ranks=[0], backend="nccl", timeout=timedelta(seconds=45)
                    )
                    x = ref.cuda()
                    dist.all_reduce(x, group=group, op=dist._make_nccl_premul_sum(2.0))
                    exact(x, ref * 2)
                    dist.destroy_process_group(group)
                    passed("subgroup-premul-destroy")
                    producer = torch.cuda.Stream()
                    consumer = torch.cuda.Stream()
                    producer.wait_stream(stream)
                    with torch.cuda.stream(producer):
                        x = ref.cuda()
                        work = dist.all_reduce(
                            x, op=dist._make_nccl_premul_sum(2.0), async_op=True
                        )
                        work.wait()
                        event = torch.cuda.Event()
                        event.record()
                    with torch.cuda.stream(consumer):
                        consumer.wait_event(event)
                        consumed = x * 4
                        exact(consumed, ref * 8)
                    stream.wait_stream(consumer)
                    passed("cross-stream-event-immediate-consumption")
                    for name, action, text in (
                        ("cpu", lambda: dist.all_reduce(torch.ones(3)), "cpu"),
                        (
                            "strided",
                            lambda: dist.all_reduce(torch.ones(6, device="cuda")[::2]),
                            "contiguous",
                        ),
                        (
                            "band",
                            lambda: dist.all_reduce(x, op=dist.ReduceOp.BAND),
                            "BAND",
                        ),
                        (
                            "bor",
                            lambda: dist.all_reduce(x, op=dist.ReduceOp.BOR),
                            "BOR",
                        ),
                        (
                            "bxor",
                            lambda: dist.all_reduce(x, op=dist.ReduceOp.BXOR),
                            "BXOR",
                        ),
                        (
                            "bool-avg",
                            lambda: dist.all_reduce(
                                torch.ones(3, device="cuda", dtype=torch.bool),
                                op=dist.ReduceOp.AVG,
                            ),
                            "boolean",
                        ),
                    ):
                        refuse("group-" + name, action, text)
                    for dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
                        refuse(
                            f"group-{dtype}",
                            lambda: dist.all_reduce(
                                torch.empty(3, device="cuda", dtype=dtype)
                            ),
                            "Unsupported Float8",
                        )
                    for dtype in (torch.bfloat16, torch.float32):
                        for bucket_view in (False, True):
                            native = torch.nn.Linear(
                                4, 4, bias=False, dtype=dtype
                            ).cuda()
                            with torch.no_grad():
                                native.weight.fill_(0.125)
                            ddp = torch.nn.parallel.DistributedDataParallel(
                                native,
                                device_ids=[0],
                                gradient_as_bucket_view=bucket_view,
                            )
                            opt = torch.optim.SGD(native.parameters(), lr=1 / 16)
                            value = torch.full((4, 4), 0.125, dtype=torch.float64)
                            for _ in range(2):
                                opt.zero_grad(set_to_none=True)
                                for micro in range(2):
                                    if micro == 0:
                                        with ddp.no_sync():
                                            out = ddp(
                                                torch.eye(4, device="cuda", dtype=dtype)
                                            )
                                            exact(out, value.T.to(dtype))
                                            out.square().sum().backward()
                                    else:
                                        out = ddp(
                                            torch.eye(4, device="cuda", dtype=dtype)
                                        )
                                        exact(out, value.T.to(dtype))
                                        out.square().sum().backward()
                                exact(native.weight.grad, (value * 4).to(dtype))
                                opt.step()
                                value = value * 0.75
                                exact(native.weight, value.to(dtype))
                            passed(
                                f"ddp-analytic-nosync-two-updates-{dtype}-bucket{bucket_view}"
                            )
                            del ddp, native, opt
            torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()
            assert not dist.is_initialized()
            passed(f"destroy-{cycle}")
        paths = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/librccl.so" in line
            }
        )
        expected = Path(
            "/home/nik/bc250work/rocm-10.2-gfx1013/rccl-b034/install-r1/lib/librccl.so"
        ).resolve()
        assert paths and all(Path(p).resolve() == expected for p in paths)
        report["loaded_libraries"] = paths
        passed("loaded-isolated-library")
        assert len(report["checks"]) == 178, len(report["checks"])
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
