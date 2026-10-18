#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Direct generic CSR SpSV/SpSM: types/indices/adjoints/scalars/analysis reuse."""

import argparse
import ctypes
import itertools
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(10132130)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    lib = ctypes.CDLL("libhipsparse.so")
    P, I, L, S = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_size_t
    PP = ctypes.POINTER(P)
    signatures = {
        "Create": [PP],
        "Destroy": [P],
        "SetStream": [P, P],
        "SetPointerMode": [P, I],
        "CreateCsr": [PP, L, L, L, P, P, P, I, I, I, I],
        "DestroySpMat": [P],
        "SpMatSetAttribute": [P, I, P, S],
        "CreateDnVec": [PP, L, P, I],
        "DestroyDnVec": [P],
        "CreateDnMat": [PP, L, L, L, P, I, I],
        "DestroyDnMat": [P],
        "SpSV_createDescr": [PP],
        "SpSV_destroyDescr": [P],
        "SpSM_createDescr": [PP],
        "SpSM_destroyDescr": [P],
        "SpSV_bufferSize": [P, I, P, P, P, P, I, I, P, ctypes.POINTER(S)],
        "SpSV_analysis": [P, I, P, P, P, P, I, I, P, P],
        "SpSV_solve": [P, I, P, P, P, P, I, I, P],
        "SpSM_bufferSize": [P, I, I, P, P, P, P, I, I, P, ctypes.POINTER(S)],
        "SpSM_analysis": [P, I, I, P, P, P, P, I, I, P, P],
        "SpSM_solve": [P, I, I, P, P, P, P, I, I, P, P],
    }
    for name, types in signatures.items():
        fn = getattr(lib, "hipsparse" + name)
        fn.argtypes = types
        fn.restype = I

    def call(name, *values):
        status = getattr(lib, "hipsparse" + name)(*values)
        assert status == 0, (name, status)

    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dtype, index, op, nrhs, device_scalar in itertools.product(
        (torch.float32, torch.float64, torch.complex64, torch.complex128),
        (torch.int32, torch.int64),
        (0, 1, 2),
        (1, 3),
        (False, True),
    ):
        resources = []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream):

                def own(create, destroy, *arguments):
                    ptr = P()
                    call(create, ctypes.byref(ptr), *arguments)
                    resources.append((destroy, ptr))
                    return ptr

                handle = own("Create", "Destroy")
                call("SetStream", handle, stream.cuda_stream)
                call("SetPointerMode", handle, int(device_scalar))
                n = 17
                dense = torch.tril(torch.randn(n, n, dtype=dtype) * 0.03)
                dense.diagonal().fill_(2)
                csr = dense.to_sparse_csr()
                offsets = csr.crow_indices().to(index).cuda()
                columns = csr.col_indices().to(index).cuda()
                values = csr.values().cuda()
                data_type = {
                    torch.float32: 0,
                    torch.float64: 1,
                    torch.complex64: 4,
                    torch.complex128: 5,
                }[dtype]
                index_type = 2 if index == torch.int32 else 3
                a = own(
                    "CreateCsr",
                    "DestroySpMat",
                    n,
                    n,
                    len(values),
                    offsets.data_ptr(),
                    columns.data_ptr(),
                    values.data_ptr(),
                    index_type,
                    index_type,
                    0,
                    data_type,
                )
                fill, diag = I(0), I(0)
                call("SpMatSetAttribute", a, 0, ctypes.byref(fill), ctypes.sizeof(fill))
                call("SpMatSetAttribute", a, 1, ctypes.byref(diag), ctypes.sizeof(diag))
                b = torch.randn(n, nrhs, dtype=dtype).mT.contiguous().cuda().mT
                scalar = 0.75 + 0.125j if dtype.is_complex else 0.75
                alpha = torch.tensor(
                    scalar, dtype=dtype, device="cuda" if device_scalar else "cpu"
                )
                storage = torch.full(
                    (nrhs, n + 2),
                    823 - 519j if dtype.is_complex else 823,
                    dtype=dtype,
                    device="cuda",
                )
                x = storage[:, 1 : n + 1].mT
                sentinel = storage[:, [0, -1]].clone()
                suffix = "SpSV" if nrhs == 1 else "SpSM"
                plan = own(suffix + "_createDescr", suffix + "_destroyDescr")
                if nrhs == 1:
                    bd = own("CreateDnVec", "DestroyDnVec", n, b.data_ptr(), data_type)
                    xd = own("CreateDnVec", "DestroyDnVec", n, x.data_ptr(), data_type)
                    arguments = [
                        handle,
                        op,
                        alpha.data_ptr(),
                        a,
                        bd,
                        xd,
                        data_type,
                        0,
                        plan,
                    ]
                else:
                    bd = own(
                        "CreateDnMat",
                        "DestroyDnMat",
                        n,
                        nrhs,
                        b.stride(1),
                        b.data_ptr(),
                        data_type,
                        1,
                    )
                    xd = own(
                        "CreateDnMat",
                        "DestroyDnMat",
                        n,
                        nrhs,
                        x.stride(1),
                        x.data_ptr(),
                        data_type,
                        1,
                    )
                    arguments = [
                        handle,
                        op,
                        0,
                        alpha.data_ptr(),
                        a,
                        bd,
                        xd,
                        data_type,
                        0,
                        plan,
                    ]
                size = S()
                call(suffix + "_bufferSize", *arguments, ctypes.byref(size))
                scratch = torch.empty(
                    max(size.value, 1), dtype=torch.uint8, device="cuda"
                )
                call(suffix + "_analysis", *arguments, scratch.data_ptr())
                matrix = dense.to(
                    torch.complex128 if dtype.is_complex else torch.float64
                )
                if op == 1:
                    matrix = matrix.mT
                if op == 2:
                    matrix = matrix.mH
                residuals = []
                for reuse in range(2):
                    if reuse:
                        b.mul_(0.5)
                    call(suffix + "_solve", *arguments, *([None] if nrhs > 1 else []))
                    expected = (
                        torch.linalg.solve_triangular(
                            matrix, b.cpu().to(matrix.dtype), upper=op != 0
                        )
                        * scalar
                    )
                    actual = x.cpu().to(matrix.dtype)
                    tol = 3e-5 if dtype in (torch.float32, torch.complex64) else 2e-12
                    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
                    residual = float(
                        (matrix @ actual - scalar * b.cpu().to(matrix.dtype)).norm()
                        / (
                            matrix.norm() * actual.norm()
                            + abs(scalar) * b.cpu().to(matrix.dtype).norm()
                        )
                    )
                    assert residual < tol, residual
                    residuals.append(residual)
                    torch.testing.assert_close(
                        storage[:, [0, -1]], sentinel, rtol=0, atol=0
                    )
                rows.append(
                    dict(
                        dtype=str(dtype),
                        index=str(index),
                        operation=op,
                        nrhs=nrhs,
                        device_scalar=device_scalar,
                        residuals=residuals,
                        analysis_reused=True,
                        nondefault_stream=True,
                        output_guards=True,
                    )
                )
                args.output.write_text(
                    json.dumps(dict(status="running", checks=rows), indent=2) + "\n"
                )
        finally:
            stream.synchronize()
            for destroy, ptr in reversed(resources):
                call(destroy, ptr)
    args.output.write_text(
        json.dumps(
            dict(status="passed", checks=rows, solve_executions=2 * len(rows)), indent=2
        )
        + "\n"
    )
    print("N213_NATIVE_PASS", len(rows), 2 * len(rows), flush=True)


if __name__ == "__main__":
    main()
