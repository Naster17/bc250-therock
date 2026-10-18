#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded public stable sort and masked-select output/gradient consumers."""

import argparse
import itertools
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(":")[0] == "gfx1013", arch
    report = dict(status="running", arch=arch, checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check(name, execute):
        try:
            details = execute()
            torch.cuda.synchronize()
            row = dict(name=name, status="passed", details=details)
        except Exception as error:
            row = dict(name=name, status="failed", error=repr(error))
        report["checks"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)

    def fixture(dtype, n, layout, operation, descending=False, special=False):
        shape = (
            (3, n * 2 + 3)
            if layout == "step"
            else (n, 3) if layout == "transpose" else (3, n)
        )
        raw = ((torch.arange(shape[0] * shape[1]).reshape(shape) * 7 % 31) - 15).to(
            dtype
        ) / 8
        if special:
            flat = raw.flatten()
            flat[::7] = float("nan")
            flat[1::11] = float("inf")
            flat[2::13] = -float("inf")
            flat[3::17] = -0.0
        cpu_leaf = raw.clone().requires_grad_(not special)
        gpu_leaf = raw.cuda().detach().requires_grad_(not special)

        def view(value):
            return (
                value[:, 1 : 1 + 2 * n : 2]
                if layout == "step"
                else value.T if layout == "transpose" else value
            )

        cpu, gpu = view(cpu_leaf), view(gpu_leaf)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        original = gpu_leaf.detach().clone()
        with torch.cuda.stream(stream):
            if operation == "sort":
                actual, indices = torch.sort(
                    gpu, dim=-1, stable=True, descending=descending
                )
                expected, reference_indices = torch.sort(
                    cpu, dim=-1, stable=True, descending=descending
                )
            else:
                actual = torch.masked_select(gpu, gpu > 0.25)
                expected = torch.masked_select(cpu, cpu > 0.25)
            if not special:
                weights = torch.linspace(
                    -0.5, 0.5, actual.numel(), dtype=dtype
                ).reshape(actual.shape)
                actual.backward(weights.cuda())
                expected.backward(weights)
            if operation == "sort":
                torch.testing.assert_close(
                    indices.cpu(), reference_indices, rtol=0, atol=0
                )
        stream.synchronize()
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=0, atol=0, equal_nan=True
        )
        if not special:
            torch.testing.assert_close(
                gpu_leaf.grad.cpu(), cpu_leaf.grad, rtol=0, atol=0
            )
        torch.testing.assert_close(
            gpu_leaf.detach(), original, rtol=0, atol=0, equal_nan=True
        )
        return dict(
            elements=actual.numel(),
            strides=gpu.stride(),
            private_stream=True,
            exact_output=True,
            exact_indices=operation == "sort",
            exact_gradient=not special,
            input_unchanged=True,
        )

    for dtype, n, layout in itertools.product(
        (torch.float32, torch.float64),
        (0, 33, 4099),
        ("contiguous", "step", "transpose"),
    ):
        for descending in (False, True):
            check(
                f"sort_{dtype}_{n}_{layout}_desc{descending}",
                lambda: fixture(dtype, n, layout, "sort", descending),
            )
        check(
            f"masked_select_{dtype}_{n}_{layout}",
            lambda: fixture(dtype, n, layout, "select"),
        )
    for dtype, n, descending in itertools.product(
        (torch.float32, torch.float64), (33, 4099), (False, True)
    ):
        check(
            f"special_sort_{dtype}_{n}_desc{descending}",
            lambda: fixture(dtype, n, "step", "sort", descending, True),
        )
    report["status"] = (
        "passed" if all(r["status"] == "passed" for r in report["checks"]) else "failed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed"


if __name__ == "__main__":
    main()
