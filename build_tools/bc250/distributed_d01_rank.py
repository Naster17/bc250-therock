#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Two physical hosts: socket collectives and GPU DDP against a global CPU reference."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    p.add_argument("--port", type=int, default=29841)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(
        status="running",
        checks=[],
        rank=a.rank,
        backend=a.backend,
        torch=torch.__version__,
    )
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        dist.init_process_group(
            a.backend,
            init_method=(
                "env://"
                if "TORCHELASTIC_RUN_ID" in os.environ
                else f"tcp://192.168.1.104:{a.port}"
            ),
            rank=a.rank,
            world_size=2,
            timeout=timedelta(seconds=90),
        )
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            x = torch.full((257,), a.rank + 1.0, device="cuda", dtype=dtype)
            dist.all_reduce(x)
            torch.cuda.synchronize()
            assert torch.equal(x.cpu(), torch.full((257,), 3.0, dtype=dtype))
            r["checks"].append("two-host-GPU-sum-" + str(dtype))
        for size in (4, 262144):
            x = torch.ones(size, device="cuda")
            dist.barrier()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(4):
                x.fill_(a.rank + 1)
                dist.all_reduce(x)
            torch.cuda.synchronize()
            r.setdefault("timing", []).append(
                dict(bytes=size * 4, seconds_per_call=(time.perf_counter() - t) / 4)
            )
            assert torch.equal(x.cpu(), torch.full((size,), 3.0))
        r["checks"].append("two-host-small-and-1MiB-reference-and-timing")
        torch.manual_seed(431)
        cpu = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)
        ).double()
        # Both references start from exactly the FP32-rounded GPU parameters.
        cpu = cpu.float().double()
        gpu = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)
        ).cuda()
        gpu.load_state_dict(cpu.state_dict())
        ddp = torch.nn.parallel.DistributedDataParallel(
            gpu, device_ids=[0], broadcast_buffers=False
        )
        opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
        refopt = torch.optim.SGD(cpu.parameters(), lr=0.01)
        data = (torch.arange(32 * 8).reshape(32, 8) % 19 - 9).float() / 16
        target = (torch.arange(32 * 4).reshape(32, 4) % 13 - 6).float() / 16
        for step in range(3):
            opt.zero_grad()
            refopt.zero_grad()
            loss = (
                (
                    ddp(data[a.rank * 16 : (a.rank + 1) * 16].cuda())
                    - target[a.rank * 16 : (a.rank + 1) * 16].cuda()
                )
                .square()
                .mean()
            )
            loss.backward()
            reference = (cpu(data.double()) - target.double()).square().mean()
            reference.backward()
            for actual, expected in zip(gpu.parameters(), cpu.parameters()):
                torch.testing.assert_close(
                    actual.grad.cpu().double(), expected.grad, atol=2e-6, rtol=2e-5
                )
            opt.step()
            refopt.step()
            for actual, expected in zip(gpu.parameters(), cpu.parameters()):
                torch.testing.assert_close(
                    actual.cpu().double(), expected, atol=2e-6, rtol=2e-5
                )
            r["checks"].append(
                "global-batch-reference-DDP-gradient-and-SGD-step-" + str(step)
            )
        dist.barrier()
        dist.destroy_process_group()
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        a.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
