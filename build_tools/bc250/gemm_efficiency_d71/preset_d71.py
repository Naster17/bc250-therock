#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D71 candidate override preset generation (pure logic, workstation-testable).

Builds a rocBLAS Tensile GEMM override CSV from the bake-off winners. The
existing release preset rows are preserved verbatim at the top so the
candidate run keeps every already-qualified tuning (the launcher's automatic
preset is skipped when a manual override path is set — the candidate must be
a superset, not a replacement).
"""

from __future__ import annotations

CSV_HEADER = (
    "transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,"
    "input_type,output_type,compute_type,solution_index"
)
TYPE_MAP = {"float16": "f16_r", "float32": "f32_r", "bfloat16": "bf16_r"}
MIN_RATIO = 1.05


def winner_rows(bench: dict, min_ratio: float = MIN_RATIO) -> list[dict]:
    rows = []
    for result in bench.get("results", []):
        if result["paired_gpu_ratio"] < min_ratio:
            continue
        layout = result["rocblas_layout"]
        args = result["args"]
        mat1 = args[1] if result["op"] == "addmm" else args[0]
        rows.append(
            {
                "transA": layout["transa"],
                "transB": layout["transb"],
                "M": layout["m"],
                "N": layout["n"],
                "batch_count": 1,
                "K": layout["k"],
                "alpha": 1,
                "beta": layout["beta"],
                "lda": layout["lda"],
                "ldb": layout["ldb"],
                "ldc": layout["ldc"],
                "input_type": TYPE_MAP[mat1["dtype"]],
                "output_type": TYPE_MAP[mat1["dtype"]],
                "compute_type": "f32_r",
                "solution_index": result["selected_candidate"],
                "ratio": result["paired_gpu_ratio"],
            }
        )
    return rows


def build_preset(bench: dict, existing_csv: str, min_ratio: float = MIN_RATIO) -> str:
    lines = [CSV_HEADER]
    for line in existing_csv.splitlines():
        line = line.strip()
        if not line or line.startswith("transA,"):
            continue
        lines.append(line)
    for row in winner_rows(bench, min_ratio):
        lines.append(
            ",".join(
                str(row[key])
                for key in (
                    "transA",
                    "transB",
                    "M",
                    "N",
                    "batch_count",
                    "K",
                    "alpha",
                    "beta",
                    "lda",
                    "ldb",
                    "ldc",
                    "input_type",
                    "output_type",
                    "compute_type",
                    "solution_index",
                )
            )
        )
    return "\n".join(lines) + "\n"
