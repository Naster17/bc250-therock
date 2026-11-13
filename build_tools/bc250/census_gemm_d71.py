#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D71 GEMM layout census wrapper.

Runs the unchanged lora-smoke entry point under a TorchDispatchMode that
records every mm/bmm/addmm call with exact sizes, strides and dtypes, then
aggregates by layout. Layout (not just logical shape) is what rocBLAS solution
selection keys on — N32 found three distinct layouts for one logical mm.

Usage (inside the qualified worker, via the guardian):
    python3 census_gemm_d71.py --config <app-config.json> [app args...]

Writes gemm-census-rank<N>.json into the app output directory. The census run
is instrumented at dispatch level and therefore SLOWER than a normal step;
only counts/layouts are claimed from it, never timing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
PROJECT = Path("/home/nik/bc250-projects/lora-smoke/project")
ENTRY = PROJECT / "train_two_host.py"
GEMM_OPS = ("mm", "bmm", "addmm")


def extract_config(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    raise ValueError("wrapper requires --config <app-config.json>")


def tensor_desc(t) -> dict:
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).replace("torch.", ""),
    }


def layout_key(op: str, descs: list[dict]) -> str:
    return json.dumps({"op": op, "args": descs}, sort_keys=True)


def aggregate_calls(calls: list[dict]) -> list[dict]:
    """Pure aggregation: group recorded calls by exact layout, count them."""
    groups: dict[str, dict] = {}
    for call in calls:
        key = layout_key(call["op"], call["args"])
        if key not in groups:
            groups[key] = {"op": call["op"], "args": call["args"], "count": 0}
        groups[key]["count"] += 1
    rows = sorted(groups.values(), key=lambda r: r["count"], reverse=True)
    total = sum(r["count"] for r in rows)
    for r in rows:
        r["share"] = r["count"] / total if total else 0.0
    return rows


def rocblas_layout(op: str, descs: list[dict]) -> dict | None:
    """Convert a torch mm/addmm layout to the rocBLAS column-major problem
    (N32 convention: operands swapped, m=n_torch, n=m_torch). bmm maps to a
    strided-batched problem. Returns None for unsupported stride patterns."""
    if op == "addmm":
        # addmm(bias, mat1, mat2)
        a, b = descs[1], descs[2]
        beta = 1
    elif op in ("mm", "bmm"):
        a, b = descs[0], descs[1]
        beta = 0
    else:
        return None
    if min(a["stride"]) != 1 or min(b["stride"]) != 1:
        return None
    m_t, k_t = a["shape"][-2], a["shape"][-1]
    k_b, n_t = b["shape"][-2], b["shape"][-1]
    if k_t != k_b:
        return None
    ta, lda = ("N", b["stride"][-2]) if b["stride"][-1] == 1 else ("T", b["stride"][-1])
    tb, ldb = ("N", a["stride"][-2]) if a["stride"][-1] == 1 else ("T", a["stride"][-1])
    batch = a["shape"][0] if op == "bmm" else 1
    return {
        "transA": ta,
        "transB": tb,
        "M": n_t,
        "N": m_t,
        "K": k_t,
        "batch_count": batch,
        "lda": lda,
        "ldb": ldb,
        "ldc": n_t,
        "beta": beta,
        "input_type": a["dtype"],
        "output_type": a["dtype"],
        "compute_type": "float32",
    }


def main() -> int:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    config_path = extract_config(sys.argv[1:])
    app_config = json.loads(config_path.read_text())
    rank = int(os.environ.get("RANK", "0"))
    output_dir = Path(app_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    calls: list[dict] = []

    class GemmCensus(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = func.overloadpacket.__name__
            if name in GEMM_OPS:
                tensors = [a for a in args if isinstance(a, torch.Tensor)]
                calls.append({"op": name, "args": [tensor_desc(t) for t in tensors]})
            return func(*args, **(kwargs or {}))

    summary = {"fixture": "census_gemm_d71", "rank": rank, "config": str(config_path)}
    sys.path.insert(0, str(PROJECT))
    sys.argv = [str(ENTRY), *sys.argv[1:]]
    exit_code = 0
    try:
        with GemmCensus():
            runpy.run_path(str(ENTRY), run_name="__main__")
    except SystemExit as exc:
        # The app entry ends in sys.exit(); capture it so the census is
        # still written (D69/D70 wrapper pattern).
        exit_code = exc.code or 0
    layouts = aggregate_calls(calls)
    for row in layouts:
        row["rocblas"] = rocblas_layout(row["op"], row["args"])
    summary.update(
        {
            "torch": torch.__version__,
            "total_gemm_calls": len(calls),
            "distinct_layouts": len(layouts),
            "layouts": layouts,
            "note": "dispatch-instrumented run; counts/layouts only, not timing",
        }
    )
    (output_dir / f"gemm-census-rank{rank}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"D71_CENSUS calls={len(calls)} layouts={len(layouts)} "
        f"top={layouts[0]['count'] if layouts else 0}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
