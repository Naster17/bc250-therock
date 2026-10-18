#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Direct Lt integer contracts plus public Torch exact integer references."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(status="running", results=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    binary = Path(
        "/home/nik/bc250work/rocm-10.2-gfx1013/build/bc250-coverage/hipblaslt_dense_int8_a08"
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    (args.output.parent / "direct.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (
        result.returncode,
        result.stdout[-1500:],
        result.stderr,
    )
    found = re.search(r"DENSE_INT8_A08_PASS cases=(\d+) refusals=(\d+)", result.stdout)
    assert found
    assert int(found[1]) == 99 and int(found[2]) == 11, found.groups()
    report["direct_cases"] = int(found[1])
    report["direct_refusals"] = int(found[2])
    save()
    torch.set_num_threads(4)
    torch.manual_seed(1808)
    for m, n, k in [
        (17, 8, 8),
        (32, 32, 32),
        (65, 24, 72),
        (128, 64, 128),
        (32, 32, 1048),
        (17, 8, 131080),
    ]:
        for layout in ["row", "column", "strided"]:
            if k > 10000 and layout != "row":
                continue

            def make(rows, cols):
                if layout == "row":
                    return torch.randint(-128, 128, (rows, cols), dtype=torch.int8)
                if layout == "column":
                    return torch.randint(-128, 128, (cols, rows), dtype=torch.int8).t()
                return torch.randint(-128, 128, (rows * 2, cols * 2), dtype=torch.int8)[
                    ::2, ::2
                ]

            a, b = make(m, k), make(k, n)
            if k >= 1048:
                a.fill_(-128)
                b.fill_(-128)
            reference = (a.long() @ b.long()).to(torch.int32)
            ga, gb = a.cuda(), b.cuda()
            if layout == "strided":
                ga = torch.empty((m * 2, k * 2), device="cuda", dtype=torch.int8)[
                    ::2, ::2
                ].copy_(a)
                gb = torch.empty((k * 2, n * 2), device="cuda", dtype=torch.int8)[
                    ::2, ::2
                ].copy_(b)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                out = torch._int_mm(ga, gb)
                buffer = torch.empty((m, n), device="cuda", dtype=torch.int32)
                returned = torch._int_mm(ga, gb, out=buffer)
            stream.synchronize()
            assert returned.data_ptr() == buffer.data_ptr()
            assert torch.equal(out.cpu(), reference) and torch.equal(
                buffer.cpu(), reference
            ), (m, n, k, layout)
            report["results"].append(
                dict(
                    status="passed",
                    m=m,
                    n=n,
                    k=k,
                    layout=layout,
                    exact_cpu_int64_reference_mod32=True,
                    custom_stream=True,
                    out_variant=True,
                )
            )
            save()
    a = torch.randint(-128, 128, (32, 32), device="cuda", dtype=torch.int8)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        out = torch._int_mm(a, a)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output.parent / "int-mm-trace.json"))
    events = prof.events()
    assert any("gemm_int8" in e.name for e in events)
    invalid = [
        lambda: torch._int_mm(a.float(), a),
        lambda: torch._int_mm(a, a, out=torch.empty((32, 32), device="cuda")),
        lambda: torch._int_mm(
            a, a, out=torch.empty((32, 64), device="cuda", dtype=torch.int32)[:, ::2]
        ),
        lambda: torch._int_mm(a[:16], a),
        lambda: torch._int_mm(a[:, :7], a[:7]),
    ]
    for fn in invalid:
        try:
            fn()
        except (RuntimeError, TypeError):
            pass
        else:
            raise AssertionError("invalid integer contract accepted")
    report.update(
        status="passed",
        torch_refusals=len(invalid),
        native_kernel_trace=True,
        torch=torch.__version__,
        arch=torch.cuda.get_device_properties(0).gcnArchName,
    )
    save()


if __name__ == "__main__":
    main()
