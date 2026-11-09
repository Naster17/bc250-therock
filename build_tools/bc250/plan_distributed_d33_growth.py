#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only larger-shape state floors; never an allocator/admission or fit proof."""
import argparse
import json
from pathlib import Path
from capacity_plan_d32 import CONFIG, PARAMETERS, GIB


def elements(width, blocks, vocab, positions):
    # Plain GPT2, tied token embedding/output head, two LayerNorms per block.
    return (
        blocks * (12 * width * width + 13 * width)
        + vocab * width
        + positions * width
        + 2 * width
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verified-results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    proof = json.loads(a.verified_results.read_text())
    assert (
        proof["status"] == "passed" and proof["fresh_CPU_reference_executed"] is False
    )
    assert (
        elements(CONFIG["width"], CONFIG["blocks"], CONFIG["vocab"], CONFIG["tokens"])
        == PARAMETERS
    )
    candidates = []
    for width, blocks, heads in [(1024, 12, 16), (1536, 12, 24), (2048, 20, 32)]:
        count = elements(width, blocks, 8192, 256)
        candidates.append(
            dict(
                width=width,
                blocks=blocks,
                heads=heads,
                vocab=8192,
                positions=256,
                parameters=count,
                single_FP32_master_gradient_AdamW_floor_GiB=count * 16 / GIB,
                two_rank_FSDP_sharded_state_floor_GiB_per_rank=count * 8 / GIB,
                two_rank_FSDP_unsharded_no_sync_gradient_floor_GiB_per_rank=count
                * 10
                / GIB,
                one_block_full_FP32_parameter_MiB=(12 * width * width + 13 * width)
                * 4
                / 1024**2,
                GPU_admitted=False,
                qualification="planning only: new shape and initialization/activation/collective envelope unqualified",
            )
        )
    result = dict(
        status="plan-only",
        baseline_parameters=PARAMETERS,
        measured_fresh_host_min_available_GiB=min(
            r["whole_host_min_available_GiB"] for r in proof["rows"]
        ),
        measured_FSDP_cgroup_max_MiB=max(
            r["measured_cgroup_max_MiB"] for r in proof["rows"] if r["mode"] == "fsdp"
        ),
        candidates=candidates,
        excludes=[
            "activations",
            "largest-unit gather overlap",
            "allocator reservations",
            "CPU initialization peak",
            "native runtime and communication allocations",
        ],
        next="Start with the smallest candidate; qualify a memory-safe initialization/shape-specific correctness path and strict admission before any larger GPU allocation. Prefer a capacity-oriented one-local-micro policy to avoid unsharded no_sync gradients; do not infer linear physical scaling from these floors.",
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
