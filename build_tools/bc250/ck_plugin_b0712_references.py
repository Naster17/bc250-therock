#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Force the private MIOpen DL solver and verify explicit refusals and coexistence."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess


def mixed_public_dtypes():
    import torch
    import torch.nn.functional as F

    lib = C.CDLL("libMIOpen.so.1")
    P = C.c_void_p
    I = C.c_int
    Z = C.c_size_t

    def api(name, types, *values):
        fn = getattr(lib, "miopen" + name)
        fn.argtypes = types
        fn.restype = I
        status = fn(*values)
        assert status == 0, (name, status)

    def storage(shape, dtype, values=None):
        count = 1
        for n in shape:
            count *= n
        size = count * torch.empty((), dtype=dtype).element_size()
        raw = torch.full((256 + 2 * size,), 0xA5, dtype=torch.uint8, device="cuda")
        view = raw[128 : 128 + size].view(dtype).reshape(shape)
        if values is not None:
            view.copy_(values.cuda())
        return raw, view, size

    def guards(raw, size):
        host = raw.cpu()
        assert bool((host[:128] == 0xA5).all()) and bool(
            (host[128 + size :] == 0xA5).all()
        )

    handle = P()
    api("Create", [C.POINTER(P)], C.byref(handle))
    api("SetStream", [P, P], handle, P(torch.cuda.current_stream().cuda_stream))
    rows = []
    try:
        for dtype in (torch.float16, torch.float32, torch.float16, torch.float32):
            descs = []
            conv = P()
            try:
                x = ((torch.arange(280).reshape(2, 4, 5, 7) % 19 - 9) / 11).to(dtype)
                w = ((torch.arange(144).reshape(4, 4, 3, 3) % 17 - 8) / 11).to(dtype)
                ref = F.conv2d(x.double(), w.double(), padding=1)
                ar, a, ab = storage(x.shape, dtype, x)
                br, b, bb = storage(w.shape, dtype, w)
                yr, y, yb = storage(ref.shape, dtype)
                input_before = ar.cpu().clone()
                weights_before = br.cpu().clone()
                for tensor in (a, b, y):
                    desc = P()
                    api("CreateTensorDescriptor", [C.POINTER(P)], C.byref(desc))
                    descs.append(desc)
                    api(
                        "Set4dTensorDescriptor",
                        [P, I, I, I, I, I],
                        desc,
                        0 if dtype == torch.float16 else 1,
                        *tensor.shape,
                    )
                xd, wd, yd = descs
                api("CreateConvolutionDescriptor", [C.POINTER(P)], C.byref(conv))
                api(
                    "InitConvolutionDescriptor",
                    [P] + [I] * 7,
                    conv,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                )
                space = Z()
                api(
                    "ConvolutionForwardGetSolutionWorkspaceSize",
                    [P] * 5 + [C.c_uint64, C.POINTER(Z)],
                    handle,
                    wd,
                    xd,
                    conv,
                    yd,
                    137,
                    C.byref(space),
                )
                assert space.value < 64 * 1024 * 1024
                work = torch.full(
                    (256 + 2 * max(space.value, 1),),
                    0xA5,
                    dtype=torch.uint8,
                    device="cuda",
                )
                api(
                    "ConvolutionForwardCompileSolution",
                    [P] * 5 + [C.c_uint64],
                    handle,
                    wd,
                    xd,
                    conv,
                    yd,
                    137,
                )
                api(
                    "ConvolutionForwardImmediate",
                    [P] * 9 + [Z, C.c_uint64],
                    handle,
                    wd,
                    P(b.data_ptr()),
                    xd,
                    P(a.data_ptr()),
                    conv,
                    yd,
                    P(y.data_ptr()),
                    P(work.data_ptr() + 128) if space.value else P(),
                    space.value,
                    137,
                )
                torch.cuda.synchronize()
                actual = y.cpu().double()
                error = (actual - ref).abs()
                tolerance = 2e-5 + 5e-5 * ref.abs()
                if dtype == torch.float16:
                    spacing = torch.maximum(
                        torch.full_like(ref, 2.0**-24),
                        torch.pow(2.0, torch.floor(torch.log2(ref.abs())) - 10),
                    )
                    tolerance = tolerance + 0.5 * spacing
                assert torch.isfinite(actual).all() and bool((error <= tolerance).all())
                guards(yr, yb)
                guards(ar, ab)
                guards(br, bb)
                guards(work, max(space.value, 1))
                assert torch.equal(ar.cpu(), input_before) and torch.equal(
                    br.cpu(), weights_before
                )
                rows.append(
                    dict(
                        dtype=str(dtype),
                        max_abs_error=float(error.max()),
                        workspace=space.value,
                    )
                )
            finally:
                if conv.value:
                    api("DestroyConvolutionDescriptor", [P], conv)
                for d in descs:
                    api("DestroyTensorDescriptor", [P], d)
    finally:
        api("Destroy", [P], handle)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    assert Path(os.environ["MIOPEN_CK_LIB_PATH"]) == root / "lib"
    r = dict(status="running", checks=[], native={})
    try:
        lib = C.CDLL(str(root / "lib/libMIOpenCKGroupedConv_gfx1013.so"))
        audit = json.loads((root / "manifest.json").read_text())
        assert all(getattr(lib, s) for s in audit["required_symbols"])
        assert lib.ck_impl_get_api_version() == 5
        fill = lib.ck_impl_fwd_get_all_kernel_type_strings
        fill.argtypes = [C.POINTER(C.c_void_p)]
        fill.restype = C.c_int
        size = lib.ck_impl_kernel_list_size
        size.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        size.restype = C.c_int
        get = lib.ck_impl_kernel_list_get
        get.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_char_p)]
        get.restype = C.c_int
        free = lib.ck_impl_kernel_list_free
        free.argtypes = [C.c_void_p]
        free.restype = None
        handle = C.c_void_p()
        assert fill(None) == 1
        assert fill(C.byref(handle)) == 0
        count = C.c_size_t()
        assert size(handle, C.byref(count)) == 0 and count.value == 9
        ids = []
        for index in range(count.value):
            value = C.c_char_p()
            assert get(handle, index, C.byref(value)) == 0
            ids.append(value.value.decode())
        value = C.c_char_p(b"unchanged")
        assert (
            get(handle, count.value, C.byref(value)) == 2
            and value.value == b"unchanged"
        )
        assert size(None, C.byref(count)) == 1
        free(handle)
        assert len(set(ids)) == 9 and all("dl" in s.lower() for s in ids)
        r["kernel_ids"] = ids
        r["checks"].append("api5-required66-symbols-and-list-contract")

        def probe(label, command, env=None):
            proc = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=240
            )
            (args.output.parent / (label + "-stdout.log")).write_text(proc.stdout)
            (args.output.parent / (label + "-stderr.log")).write_text(proc.stderr)
            proc.check_returncode()
            value = json.loads(proc.stdout.strip().splitlines()[-1])
            assert value["status"] == "passed"
            r["native"][label] = value
            r["checks"].append(label)
            return value

        old = root / "probes/probe-b072"
        value = probe("retained-public-FP32-and-unsupported-contracts", [str(old)])
        assert (
            value["executions"] == 32 and value["refusals"] == 152 and value["guards"]
        )
        value = probe(
            "public-FP16-CPU-reference", [str(root / "probes/probe-b0710_public")]
        )
        assert value["executions"] == 32 and value["refusals"] == 0 and value["guards"]
        env = os.environ.copy()
        env.pop("MIOPEN_CK_LIB_PATH", None)
        value = probe(
            "absent-private-path-FP16-refusals",
            [str(root / "probes/probe-b0710_public"), "--negative"],
            env,
        )
        assert value["executions"] == 0 and value["refusals"] == 32
        value = probe(
            "retained-FP32-per-ID-contracts",
            [
                str(root / "probes/probe-b073"),
                str(root / "lib/libMIOpenCKGroupedConv_gfx1013.so"),
            ],
        )
        assert (
            value["fixtures"] == 80
            and value["executions"] == 256
            and value["api_refusals"] == 122
            and value["runtime_refusals"] == 246
            and len(value["kernel_counts"]) == 6
        )
        value = probe(
            "FP16-nine-IDs-typed-runtime-contracts",
            [
                str(root / "probes/probe-b0710_half"),
                str(root / "lib/libMIOpenCKGroupedConv_gfx1013.so"),
            ],
        )
        assert (
            value["fixtures"] == 88
            and value["executions"] >= 200
            and value["api_refusals"] == 184
            and value["runtime_refusals"] == 477
            and len(value["kernel_counts"]) == 9
        )
        assert all(n > 0 for n in value["kernel_counts"].values()) and value["guards"]
        import torch
        import torch.nn.functional as F

        r["mixed_public_dtypes"] = mixed_public_dtypes()
        r["checks"].append(
            "same-handle-public-half-float-half-float-CPU-reference-and-guards"
        )
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        torch.set_num_threads(1)
        torch.manual_seed(7201)
        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        r["checks"].append("native-torch-backward-after-private-plugin")
        for groups in (1, 3):
            for f in (1, 3):
                values = [
                    torch.randn(2, 4 * groups, 5, 7) / 8,
                    torch.randn(4 * groups, 4, f, f) / 8,
                    torch.randn(4 * groups) / 8,
                ]
                values = [v.half() for v in values]
                cpu = [v.double().requires_grad_() for v in values]
                gpu = [v.cuda().requires_grad_() for v in values]
                a = F.conv2d(*cpu, padding=f // 2, groups=groups)
                b = F.conv2d(*gpu, padding=f // 2, groups=groups)
                cot = (
                    torch.linspace(-0.5, 0.5, a.numel(), dtype=torch.float64)
                    .reshape_as(a)
                    .half()
                )
                (a * cot.double()).sum().backward()
                (b * cot.cuda()).sum().backward()
                torch.cuda.synchronize()
                torch.testing.assert_close(b.cpu().double(), a, rtol=0.003, atol=0.002)
                for ref, actual in zip(cpu, gpu):
                    torch.testing.assert_close(
                        actual.grad.cpu().double(), ref.grad, rtol=0.006, atol=0.003
                    )
                r["checks"].append(
                    f"torch-conv-output-all-first-gradients-g{groups}-f{f}"
                )
        miopen_path = root / "runtime/libMIOpen.so.1"
        assert str(miopen_path) in Path("/proc/self/maps").read_text()
        r["mapped_miopen"] = str(miopen_path)
        r["status"] = "passed"
        assert len(r["checks"]) == 12
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
