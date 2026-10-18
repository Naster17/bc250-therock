#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Direct hipFFT plan-many with caller workspace, strides, streams and reuse."""

import argparse
import ctypes
import itertools
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(10132141)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(":")[0] == "gfx1013", arch
    library = ctypes.CDLL("libhipfft.so")
    P, I, S = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    IP = ctypes.POINTER(I)
    signatures = {
        "Create": [ctypes.POINTER(P)],
        "Destroy": [P],
        "SetAutoAllocation": [P, I],
        "MakePlanMany": [P, I, IP, IP, I, I, IP, I, I, I, I, ctypes.POINTER(S)],
        "SetWorkArea": [P, P],
        "SetStream": [P, P],
    }
    for name in ("C2C", "Z2Z", "R2C", "D2Z", "C2R", "Z2D"):
        signatures["Exec" + name] = [P, P, P] + ([I] if name in ("C2C", "Z2Z") else [])
    for name, signature in signatures.items():
        fn = getattr(library, "hipfft" + name)
        fn.argtypes, fn.restype = signature, I

    def call(name, *values):
        status = getattr(library, "hipfft" + name)(*values)
        assert status == 0, (name, status)

    report = dict(status="running", arch=arch, checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    types = [
        ("C2C", 0x29, torch.complex64),
        ("Z2Z", 0x69, torch.complex128),
        ("R2C", 0x2A, torch.float32),
        ("D2Z", 0x6A, torch.float64),
        ("C2R", 0x2C, torch.complex64),
        ("Z2D", 0x6C, torch.complex128),
    ]
    for (name, type_id, dtype), n in itertools.product(types, (17, 32)):
        directions = (-1, 1) if name in ("C2C", "Z2Z") else (None,)
        for direction in directions:
            plan = P()
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            try:
                call("Create", ctypes.byref(plan))
                call("SetAutoAllocation", plan, 0)
                inverse_real = name in ("C2R", "Z2D")
                forward_real = name in ("R2C", "D2Z")
                nin, nout = n // 2 + 1 if inverse_real else n, (
                    n // 2 + 1 if forward_real else n
                )
                outdtype = (
                    {torch.float32: torch.complex64, torch.float64: torch.complex128}[
                        dtype
                    ]
                    if forward_real
                    else (
                        {
                            torch.complex64: torch.float32,
                            torch.complex128: torch.float64,
                        }[dtype]
                        if inverse_real
                        else dtype
                    )
                )
                batch, istride, ostride, offset = 3, 2, 3, 3
                idist, odist = nin * istride + 7, nout * ostride + 11
                worksize = S()
                call(
                    "MakePlanMany",
                    plan,
                    1,
                    (I * 1)(n),
                    (I * 1)(nin),
                    istride,
                    idist,
                    (I * 1)(nout),
                    ostride,
                    odist,
                    type_id,
                    batch,
                    ctypes.byref(worksize),
                )
                workspace = torch.empty(
                    max(worksize.value, 1), dtype=torch.uint8, device="cuda"
                )
                call("SetWorkArea", plan, workspace.data_ptr())
                errors = []
                gap_writes = []
                for replay, stream in enumerate(streams):
                    stream.wait_stream(torch.cuda.current_stream())
                    call("SetStream", plan, stream.cuda_stream)
                    with torch.cuda.stream(stream):
                        source = torch.full(
                            (offset + batch * idist + 4,),
                            71.5,
                            dtype=dtype,
                            device="cuda",
                        )
                        output = torch.full(
                            (offset + batch * odist + 4,),
                            123.25,
                            dtype=outdtype,
                            device="cuda",
                        )
                        value = source.as_strided(
                            (batch, nin), (idist, istride), offset
                        )
                        result = output.as_strided(
                            (batch, nout), (odist, ostride), offset
                        )
                        data = torch.randn(batch, nin, dtype=dtype)
                        value.copy_(data)
                        extra = [] if direction is None else [direction]
                        call(
                            "Exec" + name,
                            plan,
                            value.data_ptr(),
                            result.data_ptr(),
                            *extra,
                        )
                    stream.synchronize()
                    reference = data.to(
                        torch.complex128 if dtype.is_complex else torch.float64
                    )
                    if inverse_real:
                        expected = torch.fft.irfft(reference, n=n, norm="forward")
                    elif forward_real:
                        expected = torch.fft.rfft(reference)
                    elif direction == -1:
                        expected = torch.fft.fft(reference)
                    else:
                        expected = torch.fft.ifft(reference, norm="forward")
                    tolerance = (
                        5e-4 if dtype in (torch.float32, torch.complex64) else 1e-10
                    )
                    actual = result.cpu().to(expected.dtype)
                    torch.testing.assert_close(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
                    errors.append((actual - expected).abs().max().item())
                    mask = torch.ones(output.shape, dtype=torch.bool)
                    mask.as_strided((batch, nout), (odist, ostride), offset).fill_(
                        False
                    )
                    gap_writes.append(int((output.cpu()[mask] != 123.25).sum()))
                    # rocFFT's documented noncontiguous-buffer policy permits
                    # temporary writes into stride gaps and input buffers.
                    # Guard outside the full reserved span, not those gaps.
                    for buffer, span, sentinel in (
                        (source, batch * idist, 71.5),
                        (output, batch * odist, 123.25),
                    ):
                        host = buffer.cpu()
                        assert torch.all(host[:offset] == sentinel)
                        assert torch.all(host[offset + span :] == sentinel)
                row = dict(
                    name=f"{name}_{n}_{direction}",
                    status="passed",
                    executions=2,
                    max_abs_errors=errors,
                    work_bytes=worksize.value,
                    input_stride=istride,
                    output_stride=ostride,
                    batches=batch,
                    stream_rebound=True,
                    exterior_sentinels_unchanged=True,
                    output_gap_write_counts=gap_writes,
                )
            except Exception as error:
                row = dict(
                    name=f"{name}_{n}_{direction}", status="failed", error=repr(error)
                )
            finally:
                for stream in streams:
                    stream.synchronize()
                if plan.value:
                    call("Destroy", plan)
            report["checks"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)
    report["status"] = (
        "passed" if all(r["status"] == "passed" for r in report["checks"]) else "failed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed"


if __name__ == "__main__":
    main()
