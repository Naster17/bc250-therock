#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ordinary MIOpen direct218 precision references with independently rounded operands."""
import argparse
import ctypes as C
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def public_forward(expect_old):
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
        for dtype, h, groups, filt in [
            (d, h, g, f)
            for d in (torch.float16, torch.float32)
            for h in (9, 32)
            for g, f in ((1, 3), (3, 1), (3, 3))
        ]:
            descs = []
            conv = P()
            try:
                c, k = (4, 8) if h == 9 else (16, 16)
                x = (
                    (
                        (
                            torch.arange(2 * groups * c * h * (h + 2)).reshape(
                                2, groups * c, h, h + 2
                            )
                            * 13
                            + 3
                        )
                        % 19
                        - 9
                    ).float()
                    / 11
                ).to(dtype)
                w = (
                    (
                        (
                            torch.arange(groups * k * c * filt * filt).reshape(
                                groups * k, c, filt, filt
                            )
                            * 7
                            + 5
                        )
                        % 17
                        - 8
                    ).float()
                    / 11
                ).to(dtype)
                ref = F.conv2d(x.double(), w.double(), padding=filt // 2, groups=groups)
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
                    filt // 2,
                    filt // 2,
                    1,
                    1,
                    1,
                    1,
                )
                api("SetConvolutionGroupCount", [P, I], conv, groups)
                space = Z()
                api(
                    "ConvolutionForwardGetSolutionWorkspaceSize",
                    [P] * 5 + [C.c_uint64, C.POINTER(Z)],
                    handle,
                    wd,
                    xd,
                    conv,
                    yd,
                    218,
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
                    218,
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
                    218,
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
                assert torch.isfinite(actual).all()
                failures = int((error > tolerance).sum())
                if not expect_old:
                    assert failures == 0, (
                        str(dtype),
                        h,
                        groups,
                        filt,
                        float(error.max()),
                    )
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
                        h=h,
                        groups=groups,
                        filter=filt,
                        violations=failures,
                        first_actual=float(actual.flatten()[1]),
                        first_reference=float(ref.flatten()[1]),
                        first_bound=float(tolerance.flatten()[1]),
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
    p.add_argument("--library-sha", required=True)
    p.add_argument("--expect-old", action="store_true")
    p.add_argument("--candidate-library", type=Path)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if a.candidate_library and not a.worker:
        assert (
            hashlib.sha256(a.candidate_library.read_bytes()).hexdigest()
            == a.library_sha
        )
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=str(a.candidate_library.parent)
            + ":/workspace/build/dist/rocm/lib/rocm_sysdeps/lib:"
            + os.environ.get("LD_LIBRARY_PATH", ""),
        )
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--worker"],
            env=env,
            check=True,
            timeout=430,
        )
        return
    import torch

    r = dict(status="running", checks=[])
    try:
        torch.set_num_threads(4)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        C.CDLL("libMIOpen.so.1")
        libraries = {
            Path(l.split(None, 5)[5].strip()).resolve()
            for l in Path("/proc/self/maps").read_text().splitlines()
            if len(l.split(None, 5)) == 6 and "/libMIOpen.so" in l
        }
        assert len(libraries) == 1, libraries
        library = libraries.pop()
        assert hashlib.sha256(library.read_bytes()).hexdigest() == a.library_sha
        r["library"] = str(library)
        r["library_sha256"] = a.library_sha
        rows = public_forward(a.expect_old)
        if a.expect_old:
            assert (
                rows[0]["violations"] > 0
                and abs(rows[0]["first_actual"] - 0.8427734375) < 1e-12
            )
        r["rows"] = rows
        r["checks"].append(
            "old-precision-failure-reproduced"
            if a.expect_old
            else "12-direct218-CPU-reference-guard-input-preservation-fixtures"
        )
        r["status"] = "passed"
    except BaseException as e:
        r.update(status="failed", error=repr(e))
        raise
    finally:
        a.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
