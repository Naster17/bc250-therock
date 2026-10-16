#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A02 direct reductions and recurrent-network output/gradient CPU references."""

import argparse
import copy
import ctypes as C
import json
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F


class MIOpen:
    def __init__(self):
        self.lib = C.CDLL("libMIOpen.so")

    def call(self, name, types, *values):
        f = getattr(self.lib, "miopen" + name)
        f.argtypes, f.restype = types, C.c_int
        status = f(*values)
        if status:
            raise RuntimeError(f"miopen{name}: status {status}")


def reduction(config):
    P, I, Z = C.c_void_p, C.c_int, C.c_size_t
    api = MIOpen()
    dtype = dict(f32=torch.float32, f16=torch.float16, bf16=torch.bfloat16)[
        config["dtype"]
    ]
    enum = dict(f32=1, f16=0, bf16=5)[config["dtype"]]
    shape, dims = config["shape"], tuple(config["dims"])
    source = (torch.randn(*shape) * 0.3).to(dtype)
    # Stable product reference avoids underflow and long reduction conditioning.
    if config["op"] == 1:
        source = (1 + torch.randn(*shape) * 0.005).to(dtype)
    original = source.cuda()
    if config.get("strided"):
        storage = torch.full(
            (*shape[:-1], shape[-1] * 2), 99.0, device="cuda", dtype=dtype
        )
        storage[..., ::2].copy_(original)
        original = storage[..., ::2]
    ref = source.double()
    operations = {
        0: lambda x: x.sum(dims, keepdim=True),
        1: lambda x: x.prod(dim=dims[0], keepdim=True),
        2: lambda x: x.amin(dims, keepdim=True),
        3: lambda x: x.amax(dims, keepdim=True),
        4: lambda x: x.abs().amax(dims, keepdim=True),
        5: lambda x: x.mean(dims, keepdim=True),
        6: lambda x: x.abs().sum(dims, keepdim=True),
        7: lambda x: x.square().sum(dims, keepdim=True).sqrt(),
    }
    expected = operations[config["op"]](ref)
    initial = (torch.randn_like(expected) * 0.1).to(dtype)
    guard = 32
    output_storage = torch.full(
        (initial.numel() + 2 * guard,), 19.0, device="cuda", dtype=dtype
    )
    actual = output_storage[guard:-guard].reshape(initial.shape)
    actual.copy_(initial)
    handle, reduce, ad, cd = [P() for _ in range(4)]
    created = []
    custom = config.get("custom_stream", False)
    stream = torch.cuda.Stream() if custom else torch.cuda.default_stream()
    try:
        for name, obj in [
            ("Create", handle),
            ("CreateReduceTensorDescriptor", reduce),
            ("CreateTensorDescriptor", ad),
            ("CreateTensorDescriptor", cd),
        ]:
            api.call(name, [C.POINTER(P)], C.byref(obj))
            created.append((name, obj))
        api.call("SetStream", [P, P], handle, P(stream.cuda_stream))
        for desc, t in [(ad, original), (cd, actual)]:
            sizes = (I * t.ndim)(*t.shape)
            strides = (I * t.ndim)(*t.stride())
            api.call(
                "SetTensorDescriptor",
                [P, I, I, C.POINTER(I), C.POINTER(I)],
                desc,
                enum,
                t.ndim,
                sizes,
                strides,
            )
        indexed = config.get("indices", False)
        api.call(
            "SetReduceTensorDescriptor",
            [P, I, I, I, I, I],
            reduce,
            config["op"],
            config.get("compute", 1),
            0,
            int(indexed),
            0,
        )
        ws, ix = Z(), Z()
        for name, size in [
            ("GetReductionWorkspaceSize", ws),
            ("GetReductionIndicesSize", ix),
        ]:
            api.call(
                name, [P, P, P, P, C.POINTER(Z)], handle, reduce, ad, cd, C.byref(size)
            )
        workspace_storage = torch.full(
            (max(ws.value, 1) + 512,), 165, device="cuda", dtype=torch.uint8
        )
        index_storage = torch.full(
            (max(ix.value, 1) + 512,), 165, device="cuda", dtype=torch.uint8
        )
        workspace = workspace_storage[256:-256]
        indices = index_storage[256:-256]
        alpha, beta = C.c_float(config.get("alpha", 1)), C.c_float(
            config.get("beta", 0)
        )
        torch.cuda.synchronize()
        rejected = False
        try:
            api.call(
                "ReduceTensor",
                [P, P, P, Z, P, Z, P, P, P, P, P, P],
                handle,
                reduce,
                P(indices.data_ptr()),
                ix.value,
                P(workspace.data_ptr()),
                ws.value,
                C.byref(alpha),
                ad,
                P(original.data_ptr()),
                C.byref(beta),
                cd,
                P(actual.data_ptr()),
            )
        except RuntimeError as e:
            if (
                not config.get("expect_rejection")
                or str(e) != "miopenReduceTensor: status 6"
            ):
                raise
            rejected = True
        assert rejected == config.get("expect_rejection", False)
        torch.cuda.synchronize()
        for tensor in [workspace_storage, index_storage]:
            assert bool((tensor[:256] == 165).all()) and bool(
                (tensor[-256:] == 165).all()
            )
        assert bool((output_storage[:guard] == 19).all()) and bool(
            (output_storage[-guard:] == 19).all()
        )
        if config.get("strided"):
            assert bool((storage[..., 1::2] == 99).all())
        if rejected:
            torch.testing.assert_close(actual.cpu(), initial, atol=0, rtol=0)
            return dict(rejected_status=6, output_unchanged=True)
        expected = alpha.value * expected + beta.value * initial.double()
        tol = dict(f32=4e-4, f16=0.004, bf16=0.03)[config["dtype"]]
        torch.testing.assert_close(actual.cpu().double(), expected, atol=tol, rtol=tol)
        if indexed:
            # Index fixtures reduce one dimension, without ties.
            expected_index = (
                source.double().argmin(dims[0], keepdim=True)
                if config["op"] == 2
                else source.double().argmax(dims[0], keepdim=True)
            )
            got = (
                indices[: ix.value]
                .view(torch.int32)
                .cpu()
                .reshape(expected_index.shape)
            )
            torch.testing.assert_close(got.long(), expected_index, atol=0, rtol=0)
        return dict(
            workspace=ws.value, index_bytes=ix.value, output_shape=list(expected.shape)
        )
    finally:
        destroys = {
            "Create": "Destroy",
            "CreateReduceTensorDescriptor": "DestroyReduceTensorDescriptor",
            "CreateTensorDescriptor": "DestroyTensorDescriptor",
        }
        for name, obj in reversed(created):
            api.call(destroys[name], [P], obj)


def recurrent(config):
    dtype = dict(f32=torch.float32, f16=torch.float16)[config["dtype"]]
    cls = getattr(torch.nn, config["kind"])
    model = cls(
        3,
        config["hidden"],
        num_layers=config["layers"],
        bidirectional=config["bidirectional"],
        batch_first=True,
        dropout=0,
    ).to(dtype)
    cpu = copy.deepcopy(model).double()
    gpu = copy.deepcopy(model).cuda()
    x = (torch.randn(2, config["length"], 3) * 0.2).to(dtype)
    state_shape = (
        config["layers"] * (2 if config["bidirectional"] else 1),
        2,
        config["hidden"],
    )
    states = [
        (torch.randn(state_shape) * 0.1).to(dtype)
        for _ in range(2 if config["kind"] == "LSTM" else 1)
    ]
    ci = [x.double().requires_grad_()] + [s.double().requires_grad_() for s in states]
    gi = [x.cuda().requires_grad_()] + [s.cuda().requires_grad_() for s in states]

    def run(m, inputs):
        state = tuple(inputs[1:]) if config["kind"] == "LSTM" else inputs[1]
        y, h = m(inputs[0], state)
        return [y, *h] if isinstance(h, tuple) else [y, h]

    expected, actual = run(cpu, ci), run(gpu, gi)
    tol = 4e-4 if dtype == torch.float32 else 0.008
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu().double(), b, atol=tol, rtol=tol)
    cg = [
        torch.linspace(-0.3, 0.4, a.numel()).double().reshape(a.shape) for a in expected
    ]
    torch.autograd.backward(expected, cg)
    torch.autograd.backward(actual, [g.to("cuda", dtype) for g in cg])
    count = 0
    for a, b in zip(gi + list(gpu.parameters()), ci + list(cpu.parameters())):
        torch.testing.assert_close(a.grad.cpu().double(), b.grad, atol=tol, rtol=tol)
        count += 1
    torch.cuda.synchronize()
    return dict(forward_outputs=len(actual), gradient_tensors=count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--suite", choices=["reduction", "rnn", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--collect-failures", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(251021)
    prop = torch.cuda.get_device_properties(0)
    assert prop.gcnArchName.split(":")[0] == "gfx1013"
    assert prop.warp_size == 32
    cases = []
    if args.suite in ["reduction", "all"]:
        if args.smoke:
            cases += [
                dict(suite="reduction", shape=[3, n], dims=[1], op=0, dtype="f32")
                for n in [7, 65]
            ]
        else:
            # Exercise thread, wave32, block and multiblock reductions.
            cases += [
                dict(suite="reduction", shape=[3, n], dims=[1], op=0, dtype=d)
                for n in [7, 65, 513, 4097]
                for d in ["f32", "f16", "bf16"]
            ]
            cases += [
                dict(suite="reduction", shape=[1, n], dims=[1], op=0, dtype=d)
                for n in [513, 4097]
                for d in ["f32", "f16", "bf16"]
            ]
            cases += [
                dict(suite="reduction", shape=[3, 65], dims=[1], op=op, dtype=d)
                for op in range(1, 8)
                for d in ["f32", "f16", "bf16"]
            ]
            # Nonzero beta must decode BF16 storage before arithmetic; non-unit
            # scalars must retain the API's FP32 compute precision.
            cases += [
                dict(
                    suite="reduction",
                    shape=[3, n],
                    dims=[1],
                    op=0,
                    dtype=d,
                    alpha=-0.73123,
                    beta=0.25123,
                )
                for n in [7, 65, 513, 4097]
                for d in ["f32", "f16", "bf16"]
            ]
            cases += [
                dict(
                    suite="reduction",
                    shape=[3, 17, 5],
                    dims=[1],
                    op=0,
                    dtype="f32",
                    strided=True,
                    custom_stream=True,
                    alpha=-0.75,
                    beta=0.25,
                ),
                dict(
                    suite="reduction",
                    shape=[3, 65],
                    dims=[1],
                    op=2,
                    dtype="f32",
                    indices=True,
                ),
                dict(
                    suite="reduction",
                    shape=[3, 65],
                    dims=[1],
                    op=3,
                    dtype="f32",
                    indices=True,
                ),
            ]
        if not args.smoke:
            cases += [
                dict(
                    suite="reduction",
                    shape=[3, 7],
                    dims=[1],
                    op=0,
                    dtype="bf16",
                    compute=c,
                    expect_rejection=True,
                )
                for c in [0, 5, 6]
            ]
    if args.suite in ["rnn", "all"]:
        configs = (
            [("LSTM", "f32", 1, False, 4, 5)]
            if args.smoke
            else [
                (k, d, l, b, h, t)
                for k in ["LSTM", "GRU", "RNN"]
                for d in ["f32", "f16"]
                for l, b, h, t in [(1, False, 4, 5), (2, True, 7, 9)]
            ]
        )
        cases += [
            dict(
                suite="rnn",
                kind=k,
                dtype=d,
                layers=l,
                bidirectional=b,
                hidden=h,
                length=t,
            )
            for k, d, l, b, h, t in configs
        ]
    rows = []
    for config in cases:
        row = dict(config)
        try:
            row.update(
                (reduction if config["suite"] == "reduction" else recurrent)(config)
            )
            row["status"] = "passed"
        except Exception as e:
            row.update(status="failed", error=str(e))
            traceback.print_exc()
        rows.append(row)
        out = dict(status="running", arch=prop.gcnArchName, results=rows)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if row["status"] != "passed" and not args.collect_failures:
            break
    out.update(
        status=(
            "passed"
            if all(r["status"] == "passed" for r in rows) and len(rows) == len(cases)
            else "failed"
        )
    )
    out["loaded_libraries"] = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(
                n in line
                for n in ["libMIOpen.so", "libhsa-runtime64.so", "libtorch_hip.so"]
            )
        }
    )
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    if not args.collect_failures:
        assert out["status"] == "passed"


if __name__ == "__main__":
    main()
