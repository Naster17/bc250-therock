#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Native sentinel visibility plus CPU-first profiling and FP64 GPU references."""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def hsa_uuid():
    class Agent(ctypes.Structure):
        _fields_ = [("handle", ctypes.c_uint64)]

    library = ctypes.CDLL("/workspace/build/dist/rocm/lib/libhsa-runtime64.so")
    library.hsa_agent_get_info.argtypes = [Agent, ctypes.c_int, ctypes.c_void_p]
    library.hsa_agent_get_info.restype = ctypes.c_int
    values = []
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, Agent, ctypes.c_void_p)

    @callback_type
    def callback(agent, unused):
        device = ctypes.c_uint32()
        if (
            library.hsa_agent_get_info(agent, 17, ctypes.byref(device)) == 0
            and device.value == 1
        ):
            uuid = ctypes.create_string_buffer(21)
            if library.hsa_agent_get_info(agent, 0xA011, uuid) == 0:
                values.append(uuid.value.decode())
        return 0

    library.hsa_iterate_agents.argtypes = [callback_type, ctypes.c_void_p]
    library.hsa_iterate_agents.restype = ctypes.c_int
    assert library.hsa_iterate_agents(callback, None) == 0
    assert len(values) == 1, values
    if values[0] != "GPU-XX":
        assert values[0].startswith("GPU-") and len(values[0]) == 20, values
        int(values[0][4:], 16)
    return values[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-missing-trace", action="store_true")
    parser.add_argument("--sdk-sha", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    paths = {
        line.split(maxsplit=5)[5].strip()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "librocprofiler-sdk.so" in line
    }
    assert len(paths) == 1, paths
    loaded = paths.pop()
    assert hashlib.sha256(Path(loaded).read_bytes()).hexdigest() == args.sdk_sha
    control = args.sdk_sha.startswith("a6dc1c88")
    assert loaded.startswith("/workspace/build/dist/rocm/lib/"), loaded
    assert not torch.cuda.is_initialized()
    captures, checks = [], []

    def capture(name, gpu, function):
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if gpu else [])
        with profile(activities=activities, profile_memory=True) as prof:
            value = function()
        trace = args.output.parent / (name + ".json")
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        copies = [e for e in events if e.get("cat") == "gpu_memcpy"]
        runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "hip_runtime")]
        assert any(e.get("cat") == "cpu_op" for e in events)
        if gpu and args.expect_missing_trace:
            assert not kernels and not copies
        elif gpu:
            assert kernels and len(copies) >= 2
            assert all(
                math.isfinite(e["dur"]) and e["dur"] > 0 for e in kernels + copies
            )
            identities = lambda values: {
                e.get("args", {}).get("correlation") for e in values
            } - {None}
            assert identities(kernels) & identities(runtime)
            assert identities(copies) & identities(runtime)
            assert any(e.kernels for e in prof.events())
        else:
            assert not kernels and not copies
        captures.append(
            dict(
                name=name,
                kernels=len(kernels),
                copies=len(copies),
                runtime=len(runtime),
            )
        )
        return value

    assert capture("cpu-cold", False, lambda: torch.arange(33).sum().item()) == 528
    checks.append("cpu-sum")
    assert not torch.cuda.is_initialized()
    count = torch.cuda.device_count()
    result = dict(
        status="running",
        kind=args.kind,
        sdk_path=loaded,
        sdk_sha256=args.sdk_sha,
        device_count=count,
        control=control,
        expected_missing_trace=args.expect_missing_trace,
        captures=captures,
        checks=checks,
    )
    if args.kind == "cpu":
        assert (
            count == 0
            and not torch.cuda.is_available()
            and not torch.cuda.is_initialized()
        )
    else:
        assert count == 1
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        generator = torch.Generator().manual_seed(4800)
        cx = (torch.randn(32, 48, generator=generator) * 0.1).double().requires_grad_()
        cw = (torch.randn(48, 24, generator=generator) * 0.1).double().requires_grad_()
        reference = (cx @ cw).sin()
        ref_gradients = torch.autograd.grad(reference.square().mean(), (cx, cw))
        x = cx.detach().float().cuda().requires_grad_()
        w = cw.detach().float().cuda().requires_grad_()
        host = torch.arange(65536, dtype=torch.float32).pin_memory()
        target = torch.empty_like(host, device="cuda")
        back = torch.empty_like(host, pin_memory=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def work(label):
            with torch.cuda.stream(stream):
                y = (x @ w).sin()
                dx, dw = torch.autograd.grad(y.square().mean(), (x, w))
                target.copy_(host, non_blocking=True)
                back.copy_(target, non_blocking=True)
            stream.synchronize()
            torch.testing.assert_close(back, host, rtol=0, atol=0)
            checks.append("copy-" + label)
            return y, dx, dw

        plain = work("plain")
        for name, actual, expected in zip(
            ("output", "dx", "dw"), plain, (reference, *ref_gradients)
        ):
            torch.testing.assert_close(
                actual.detach().cpu().double(), expected, atol=2e-6, rtol=2e-4
            )
            checks.append("fp64-" + name)
        for index in range(2):
            actual = capture("gpu-" + str(index), True, lambda: work(str(index)))
            for name, got, expected in zip(("output", "dx", "dw"), actual, plain):
                torch.testing.assert_close(got, expected, atol=0, rtol=0)
                checks.append(f"profile{index}-" + name)
        result.update(
            uuid=hsa_uuid(),
            digests=[digest(t) for t in plain],
            architecture="gfx1013:xnack-",
            allocated_peak=torch.cuda.max_memory_allocated(),
            reserved_peak=torch.cuda.max_memory_reserved(),
        )
    result["status"] = "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(status="passed", kind=args.kind, count=count, checks=len(checks))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
