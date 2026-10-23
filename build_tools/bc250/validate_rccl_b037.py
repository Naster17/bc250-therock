#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Filtered gfx1013 RCCL: numeric FP8 refusals, bit copies and memory API readiness."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(output, packaged=False):
    import ctypes
    import torch
    import torch.distributed as dist
    from torch.cuda import nccl

    report = dict(status="running", checks=[], refusals={}, world_size=1)

    def passed(name):
        report["checks"].append(name)
        output.write_text(json.dumps(report, indent=2) + "\n")

    def exact(actual, reference):
        assert actual.device.type == "cuda" and reference.device.type == "cpu"
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    def refuse(name, action, text):
        try:
            action()
        except (RuntimeError, ValueError, TypeError) as error:
            assert text.lower() in str(error).lower(), (name, str(error))
            report["refusals"][name] = str(error)
        else:
            raise AssertionError("Expected refusal: " + name)
        passed(name)

    def copy_cases(scope, stream):
        for dtype in copy_types:
            for count in (17, 257):
                bits = (torch.arange(count, dtype=torch.int64) % 256).to(torch.uint8)
                raw = torch.full((count + 10,), 19, dtype=torch.uint8, device="cuda")
                x = raw[5:-5].view(dtype)
                x.view(torch.uint8).copy_(bits)
                if scope == "lowlevel":
                    nccl.broadcast([x], streams=[stream])
                else:
                    dist.broadcast(x, src=0)
                exact(x.view(torch.uint8), bits)
                exact(raw[:5], torch.full((5,), 19, dtype=torch.uint8))
                exact(raw[-5:], torch.full((5,), 19, dtype=torch.uint8))
                passed(f"{scope}-broadcast-bits-{dtype}-{count}")
                target = torch.full_like(raw, 23)
                y = target[5:-5].view(dtype)
                if scope == "lowlevel":
                    nccl.all_gather([x], [y], streams=[stream])
                else:
                    dist.all_gather([y], x)
                exact(y.view(torch.uint8), bits)
                exact(x.view(torch.uint8), bits)
                exact(target[:5], torch.full((5,), 23, dtype=torch.uint8))
                exact(target[-5:], torch.full((5,), 23, dtype=torch.uint8))
                passed(f"{scope}-gather-bits-{dtype}-{count}")

    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        assert torch.cuda.device_count() == 1
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
        assert dist.is_nccl_available() and nccl.version() == (2, 32, 3)
        passed("device-and-backend")
        copy_types = (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        )
        numeric_types = copy_types + (torch.float8_e8m0fnu,)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for dtype in numeric_types:
                x = torch.empty(17, device="cuda", dtype=dtype)
                y = torch.empty_like(x)
                for name, action in (
                    ("allreduce", lambda: nccl.all_reduce([x], [y])),
                    ("reduce", lambda: nccl.reduce([x], y)),
                    ("reduce-scatter", lambda: nccl.reduce_scatter([x], [y])),
                ):
                    refuse(
                        f"lowlevel-{name}-{dtype}",
                        action,
                        "numeric reductions do not support Float8",
                    )
            copy_cases("lowlevel", stream)
            dist.init_process_group(
                "nccl",
                init_method="file://" + str(output.parent / "owned-store"),
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=45),
                device_id=torch.device("cuda:0"),
            )
            for dtype in numeric_types:
                x = torch.empty(17, device="cuda", dtype=dtype)
                y = torch.empty_like(x)
                for name, action in (
                    ("allreduce", lambda: dist.all_reduce(x)),
                    ("reduce", lambda: dist.reduce(x, dst=0)),
                    ("reduce-scatter", lambda: dist.reduce_scatter_tensor(y, x)),
                    ("coalesced", lambda: dist.all_reduce_coalesced([x])),
                ):
                    refuse(f"group-{name}-{dtype}", action, "Unsupported Float8")
            copy_cases("group", stream)
            backend = dist.distributed_c10d._get_default_group()._get_backend(
                torch.device("cuda:0")
            )
            assert not backend.supports_tensor_alloc(torch.device("cuda:0"))
            passed("multicast-allocator-capability-false")
            refuse(
                "multicast-allocator-refused",
                lambda: backend.mem_allocator,
                "not supported",
            )
            pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(pool):
                tensor = torch.arange(17, device="cuda", dtype=torch.float32) / 4 + 0.25
            backend.register_mem_pool(pool)
            try:
                dist.all_reduce(tensor, op=dist._make_nccl_premul_sum(2.0))
                exact(tensor, (torch.arange(17, dtype=torch.float32) / 4 + 0.25) * 2)
                with torch.cuda.use_mem_pool(pool):
                    later = (
                        torch.arange(33, device="cuda", dtype=torch.float32) / 4 + 0.25
                    )
                dist.all_reduce(later, op=dist._make_nccl_premul_sum(2.0))
                exact(later, (torch.arange(33, dtype=torch.float32) / 4 + 0.25) * 2)
            finally:
                torch.cuda.synchronize()
                backend.deregister_mem_pool(pool)
            passed("generic-pool-register-new-allocation-premul-deregister")
            del tensor, later, pool
            libpath = Path(
                "/workspace/rccl/lib/librccl.so"
                if packaged
                else "/home/nik/bc250work/rocm-10.2-gfx1013/rccl-b034/install-r1/lib/librccl.so"
            )
            library = ctypes.CDLL(str(libpath))
            register = library.ncclCommWindowRegister
            register.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
            ]
            register.restype = ctypes.c_int
            buffer = torch.ones(17, device="cuda")
            handle = ctypes.c_void_p(1)
            assert os.environ["NCCL_WIN_ENABLE"] == "0"
            assert (
                register(
                    backend._comm_ptr(),
                    buffer.data_ptr(),
                    buffer.numel() * buffer.element_size(),
                    ctypes.byref(handle),
                    1,
                )
                == 0
            )
            assert handle.value is None
            exact(buffer, torch.ones(17))
            passed("symmetric-window-optout-success-null-handle")
            torch.cuda.synchronize()
            stats = backend.memory_stats()
            assert set(stats) == {"total", "persist", "suspend", "suspended"}
            assert stats["suspended"] == 0 and all(v >= 0 for v in stats.values())
            report["memory_before"] = stats
            passed("memory-stats-readiness")
            backend.suspend()
            try:
                suspended = backend.memory_stats()
                assert suspended["suspended"] == 1
                report["memory_suspended"] = suspended
            finally:
                backend.resume()
            resumed = backend.memory_stats()
            assert resumed["suspended"] == 0
            report["memory_resumed"] = resumed
            passed("communicator-suspend-resume-state")
            ref = torch.arange(257, dtype=torch.float32) / 4
            x = ref.cuda()
            dist.all_reduce(x, op=dist._make_nccl_premul_sum(2.0))
            exact(x, ref * 2)
            passed("post-resume-premul-cpu-reference")
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()
        paths = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/librccl.so" in line
            }
        )
        assert paths and all(Path(p).resolve() == libpath.resolve() for p in paths)
        report["loaded_libraries"] = paths
        passed("loaded-isolated-dso-lifecycle")
        assert len(report["checks"]) == 76, len(report["checks"])
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
    parser.add_argument(
        "--packaged",
        action="store_true",
        help="Use ordinary launcher configuration without test overrides",
    )
    args = parser.parse_args()
    if args.packaged:
        worker(args.output, packaged=True)
        return
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
        NCCL_WIN_ENABLE="0",
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
