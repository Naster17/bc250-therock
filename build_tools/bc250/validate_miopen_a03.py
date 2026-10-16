#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A03 collect/replay genuine MIOpen solver records against CPU-double references."""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys


def probe(config):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    torch.manual_seed(251017)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    lib = C.CDLL("libMIOpen.so")
    P, I, Z = C.c_void_p, C.c_int, C.c_size_t

    class Perf(C.Structure):
        _fields_ = [("algorithm", I), ("time", C.c_float), ("memory", Z)]

    class Solution(C.Structure):
        _fields_ = [
            ("time", C.c_float),
            ("memory", Z),
            ("id", C.c_uint64),
            ("algorithm", I),
        ]

    def api(name, types, *values):
        fn = getattr(lib, "miopen" + name)
        fn.restype = I
        fn.argtypes = types
        status = fn(*values)
        if status != 0:
            raise RuntimeError(f"miopen{name}: status {status}")

    handles = []
    tensors = []
    conv = P()
    handle = P()
    row = dict(config)
    stream = (
        torch.cuda.Stream()
        if config["stream"] == "custom"
        else torch.cuda.default_stream()
    )
    dtype = dict(f32=torch.float32, f16=torch.float16, bf16=torch.bfloat16)[
        config["dtype"]
    ]
    enum = dict(f32=1, f16=0, bf16=5)[config["dtype"]]
    x = (torch.randn(*config["input"]) * 0.2).to(dtype).double().requires_grad_()
    w = (torch.randn(*config["weight"]) * 0.2).to(dtype).double().requires_grad_()
    y = F.conv2d(
        x,
        w,
        stride=config["stride"],
        padding=config["padding"],
        dilation=config["dilation"],
        groups=config["groups"],
    )
    dy = torch.linspace(-0.4, 0.6, y.numel()).reshape(y.shape).to(dtype).double()
    y.backward(dy)
    references = {"fwd": y.detach(), "dx": x.grad, "dw": w.grad}
    device = {
        "x": x.detach().to(dtype).cuda(),
        "w": w.detach().to(dtype).cuda(),
        "dy": dy.to(dtype).cuda(),
    }
    direction = config["direction"]
    output = torch.empty_like(references[direction], dtype=dtype, device="cuda")
    pairs = {"fwd": ("x", "w"), "dx": ("dy", "w"), "dw": ("dy", "x")}
    a, b = [device[k] for k in pairs[direction]]
    ds = []
    try:
        api("Create", [C.POINTER(P)], C.byref(handle))
        handles.append(handle)
        api("SetStream", [P, P], handle, P(stream.cuda_stream))
        api("EnableProfiling", [P, C.c_bool], handle, config["profiling"])
        for t in [a, b, output]:
            desc = P()
            api("CreateTensorDescriptor", [C.POINTER(P)], C.byref(desc))
            tensors.append(desc)
            api("Set4dTensorDescriptor", [P, I, I, I, I, I], desc, enum, *t.shape)
            ds.append(desc)
        ad, bd, od = ds
        api("CreateConvolutionDescriptor", [C.POINTER(P)], C.byref(conv))
        api(
            "InitConvolutionDescriptor",
            [P] + [I] * 7,
            conv,
            0,
            *config["padding"],
            *config["stride"],
            *config["dilation"],
        )
        api("SetConvolutionGroupCount", [P, I], conv, config["groups"])
        prefix = {"fwd": "Forward", "dx": "BackwardData", "dw": "BackwardWeights"}[
            direction
        ]
        size = Z()
        query = [bd, ad, conv, od] if direction == "fwd" else [ad, bd, conv, od]
        api(
            "Convolution" + prefix + "GetWorkSpaceSize",
            [P] * 5 + [C.POINTER(Z)],
            handle,
            *query,
            C.byref(size),
        )
        row["queried_workspace"] = size.value
        allocated = (
            size.value
            if config["workspace"] == "queried"
            else (max(size.value, 4096) if config["workspace"] == "extra" else 0)
        )
        workspace = (
            torch.empty(allocated, dtype=torch.uint8, device="cuda")
            if allocated
            else None
        )
        ptr = P(workspace.data_ptr()) if workspace is not None else P()
        row["provided_workspace"] = allocated
        if config["phase"] == "collect":
            count = I()
            perf = (Perf * 8)()
            torch.cuda.synchronize()
            api(
                "FindConvolution" + prefix + "Algorithm",
                [P] * 8 + [I, C.POINTER(I), C.POINTER(Perf), P, Z, C.c_bool],
                handle,
                ad,
                P(a.data_ptr()),
                bd,
                P(b.data_ptr()),
                conv,
                od,
                P(output.data_ptr()),
                8,
                C.byref(count),
                perf,
                ptr,
                allocated,
                False,
            )
            assert count.value > 0
            row["algorithms"] = [
                dict(algorithm=p.algorithm, time_ms=p.time, workspace=p.memory)
                for p in perf[: count.value]
            ]
        count = Z()
        solutions = (Solution * 128)()
        api(
            "Convolution" + prefix + "GetSolution",
            [P] * 5 + [Z, C.POINTER(Z), C.POINTER(Solution)],
            handle,
            *query,
            128,
            C.byref(count),
            solutions,
        )
        assert 0 < count.value <= 128
        row["solutions"] = []
        for sol in solutions[: count.value]:
            assert sol.time > 0
            work = torch.empty(max(sol.memory, 1), dtype=torch.uint8, device="cuda")
            output.fill_(float("nan"))
            ia, ib = (b, a) if direction == "fwd" else (a, b)
            iad, ibd = (bd, ad) if direction == "fwd" else (ad, bd)
            api(
                "Convolution" + prefix + "Immediate",
                [P] * 9 + [Z, C.c_uint64],
                handle,
                iad,
                P(ia.data_ptr()),
                ibd,
                P(ib.data_ptr()),
                conv,
                od,
                P(output.data_ptr()),
                P(work.data_ptr()),
                sol.memory,
                sol.id,
            )
            torch.cuda.synchronize()
            tol = dict(f32=4e-4, f16=0.006, bf16=0.04)[config["dtype"]]
            torch.testing.assert_close(
                output.cpu().double(), references[direction], atol=tol, rtol=tol
            )
            row["solutions"].append(
                dict(
                    id=sol.id,
                    algorithm=sol.algorithm,
                    time_ms=sol.time,
                    workspace=sol.memory,
                    max_abs_error=(output.cpu().double() - references[direction])
                    .abs()
                    .max()
                    .item(),
                    status="passed",
                )
            )
        row["phase"] = "reference_passed"
        row["status"] = "passed"
    except Exception as e:
        row.update(status="failed", error=str(e))
    finally:
        if conv.value:
            api("DestroyConvolutionDescriptor", [P], conv)
        for desc in tensors:
            api("DestroyTensorDescriptor", [P], desc)
        for h in handles:
            api("Destroy", [P], h)
    row["loaded_libraries"] = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(
                name in line
                for name in ["libMIOpen.so", "libhsa-runtime64.so", "libamdhip64.so"]
            )
        }
    )
    return row


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--phase", choices=["handle", "collect", "replay"], default="collect"
    )
    ap.add_argument(
        "--profile", choices=["all", "basic", "tune", "unseen"], default="all"
    )
    ap.add_argument("--direction", choices=["all", "fwd", "dx", "dw"], default="all")
    args = ap.parse_args()
    torch.set_num_threads(2)
    prop = torch.cuda.get_device_properties(0)
    assert prop.gcnArchName.split(":")[0] == "gfx1013" and prop.warp_size == 32
    rows = []
    if args.phase == "handle":
        lib = C.CDLL("libMIOpen.so")
        h = C.c_void_p()
        lib.miopenCreate.argtypes = [C.POINTER(C.c_void_p)]
        lib.miopenDestroy.argtypes = [C.c_void_p]
        assert lib.miopenCreate(C.byref(h)) == 0
        assert lib.miopenDestroy(h) == 0
        rows = [dict(status="passed", check="native-handle-create-destroy")]
    else:
        common = dict(
            stream="default",
            profiling=False,
            workspace="queried",
            phase=args.phase,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
        )
        profiles = [
            dict(name="basic-" + d, input=[2, 4, 13, 11], weight=[8, 4, 3, 3], dtype=d)
            for d in ["f32", "f16", "bf16"]
        ]
        profiles += [
            dict(
                name="group2",
                input=[2, 8, 9, 7],
                weight=[8, 4, 3, 3],
                dtype="f32",
                groups=2,
            ),
            dict(
                name="dilated",
                input=[2, 4, 1, 19],
                weight=[6, 4, 1, 3],
                dtype="f32",
                padding=[0, 2],
                stride=[1, 2],
                dilation=[1, 2],
            ),
        ]
        if args.profile == "basic":
            profiles = profiles[:1]
        if args.profile == "tune":
            profiles = [
                dict(name="tune", input=[1, 2, 5, 5], weight=[2, 2, 3, 3], dtype="f32")
            ]
        if args.profile == "unseen":
            profiles = [
                dict(
                    name="unseen", input=[1, 3, 7, 9], weight=[5, 3, 3, 3], dtype="f32"
                )
            ]
        for profile in profiles:
            for direction in (
                ["fwd", "dx", "dw"] if args.direction == "all" else [args.direction]
            ):
                config = dict(common, **profile, direction=direction)
                row = probe(config)
                rows.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(
                    json.dumps(dict(status="running", results=rows), indent=2) + "\n"
                )
                if row["status"] != "passed":
                    break
            if rows[-1]["status"] != "passed":
                break
    result = dict(
        status="passed" if all(r["status"] == "passed" for r in rows) else "failed",
        arch=prop.gcnArchName,
        multiprocessors=prop.multi_processor_count,
        phase=args.phase,
        results=rows,
        environment={k: v for k, v in os.environ.items() if k.startswith("MIOPEN_")},
        loaded_libraries=sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(
                    n in line
                    for n in ["libMIOpen.so", "libhsa-runtime64.so", "libtorch_hip.so"]
                )
            }
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
