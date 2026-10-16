#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Direct MIOpen convolution API isolation with independent CPU-double references."""

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
    x = (torch.randn(2, 4, 1, 19) * 0.2).double().requires_grad_()
    w = (torch.randn(6, 4, 1, 3) * 0.2).double().requires_grad_()
    y = F.conv2d(x, w, stride=(1, 2), padding=(0, 2), dilation=(1, 2))
    dy = torch.linspace(-0.4, 0.6, y.numel()).double().reshape(y.shape)
    y.backward(dy)
    references = {"fwd": y.detach(), "dx": x.grad, "dw": w.grad}
    device = {
        "x": x.detach().float().cuda(),
        "w": w.detach().float().cuda(),
        "dy": dy.float().cuda(),
    }
    direction = config["direction"]
    output = torch.empty_like(references[direction], dtype=torch.float32, device="cuda")
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
            api("Set4dTensorDescriptor", [P, I, I, I, I, I], desc, 1, *t.shape)
            ds.append(desc)
        ad, bd, od = ds
        api("CreateConvolutionDescriptor", [C.POINTER(P)], C.byref(conv))
        api("InitConvolutionDescriptor", [P] + [I] * 7, conv, 0, 0, 2, 1, 2, 1, 2)
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
        count = I()
        perf = (Perf * 8)()
        torch.cuda.synchronize()
        row["phase"] = "find"
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
        row["algorithms"] = [
            {"algorithm": p.algorithm, "time_ms": p.time, "workspace": p.memory}
            for p in perf[: count.value]
        ]
        assert count.value > 0
        row["phase"] = "execute"
        alpha, beta = C.c_float(1), C.c_float(0)
        for p in perf[: count.value]:
            assert p.time > 0 and p.memory <= allocated
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            api(
                "Convolution" + prefix,
                [P] * 7 + [I] + [P] * 4 + [Z],
                handle,
                C.byref(alpha),
                ad,
                P(a.data_ptr()),
                bd,
                P(b.data_ptr()),
                conv,
                p.algorithm,
                C.byref(beta),
                od,
                P(output.data_ptr()),
                ptr,
                allocated,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output.cpu().double(), references[direction], atol=4e-4, rtol=4e-4
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--child")
    ap.add_argument("--expect-pass", action="store_true")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--bracket-events", action="store_true")
    ap.add_argument("--child-env", action="append", default=[])
    args = ap.parse_args()
    if args.child:
        result = probe(json.loads(args.child))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    configs = []
    for direction in ["fwd", "dx", "dw"]:
        for workspace in ["queried", "extra"]:
            configs.append(
                dict(
                    direction=direction,
                    workspace=workspace,
                    stream="default",
                    profiling=False,
                )
            )
    if args.extended:
        for direction in ["fwd", "dx", "dw"]:
            for stream, profiling in [
                ("custom", False),
                ("custom", True),
                ("default", True),
            ]:
                configs.append(
                    dict(
                        direction=direction,
                        workspace="queried",
                        stream=stream,
                        profiling=profiling,
                    )
                )
    rows = []
    for index, config in enumerate(configs):
        out = args.output.parent / f"direct-{index}.json"
        with (args.output.parent / f"direct-{index}.log").open("w") as log:
            r = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--output",
                    str(out),
                    "--child",
                    json.dumps(config),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=80,
                env=dict(
                    os.environ,
                    MIOPEN_ENABLE_LOGGING="1",
                    MIOPEN_LOG_LEVEL="6",
                    **({"LD_PRELOAD": str(args.trace)} if args.trace else {}),
                    **(
                        {"BC250_A01_BRACKET_EVENTS": "1"} if args.bracket_events else {}
                    ),
                    **dict(x.split("=", 1) for x in args.child_env),
                ),
            )
        result = (
            json.loads(out.read_text())
            if out.exists()
            else dict(config, status="process_failed", returncode=r.returncode)
        )
        rows.append(result)
        args.output.write_text(json.dumps({"results": rows}, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if args.expect_pass:
        assert all(r["status"] == "passed" for r in rows)


if __name__ == "__main__":
    main()
