#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Capture both actual vocabulary GEMM layouts with owned BF16 CPU fixtures."""

import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
import validate_qwen35_training_model as existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shapes = {(1024, 2048, 248320): "forward", (1024, 248320, 2048): "input-backward"}
    preset = Path("/workspace/rocblas-gemm-overrides.csv")
    assert os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH") == str(preset)
    assert "LD_PRELOAD" not in os.environ
    os.environ["ROCBLAS_LAYER"] = "2"
    os.environ["ROCBLAS_LOG_BENCH_PATH"] = str(args.output.parent / "rocblas-bench.log")
    torch.manual_seed(31932)
    torch.set_num_threads(4)
    app_path = Path(__file__).with_name("n38_qualified_bc250_ops.py")
    spec = importlib.util.spec_from_file_location("n32_app", app_path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    model = existing.make_model(True, torch.bfloat16).train()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.001)
    app.install(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ids = torch.randint(3, model.config.vocab_size, (1, 1024), device="cuda")
    rows = {}
    counts = collections.Counter()
    stage = "forward"

    class Audit(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.mm.default:
                a, b = args
                logical = (a.shape[0], a.shape[1], b.shape[1])
                if logical in shapes and a.shape[1] == b.shape[0]:
                    metadata = dict(
                        role=shapes[logical],
                        a_shape=list(a.shape),
                        b_shape=list(b.shape),
                        a_stride=list(a.stride()),
                        b_stride=list(b.stride()),
                        a_offset=a.storage_offset(),
                        b_offset=b.storage_offset(),
                        dtype=str(a.dtype),
                    )
                    key = json.dumps(metadata, sort_keys=True)
                    counts[(key, stage)] += 1
                    if key not in rows:
                        name = (
                            "fixture-"
                            + hashlib.sha256(key.encode()).hexdigest()[:12]
                            + ".pt"
                        )
                        path = args_output.parent / name
                        torch.save(dict(a=a.detach().cpu(), b=b.detach().cpu()), path)
                        rows[key] = dict(
                            **metadata,
                            fixture=name,
                            fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                        )
            return func(*args, **(kwargs or {}))

    args_output = args.output
    with Audit(), sdpa_kernel(SDPBackend.MATH):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        stage = "backward"
        loss.backward()
    torch.cuda.synchronize()
    for key, row in rows.items():
        row["forward_mm_calls"] = counts[(key, "forward")]
        row["backward_mm_calls"] = counts[(key, "backward")]
    assert {r["role"] for r in rows.values()} == set(shapes.values())
    assert rows and all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
        if p.requires_grad
    )
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                checks=["actual-layouts", "finite-model-backward"],
                loss=float(loss.detach()),
                layouts=list(rows.values()),
                audited_logical_shapes=[list(s) for s in shapes],
                gemm_preset_sha256=hashlib.sha256(preset.read_bytes()).hexdigest(),
                note="CPU copies/saving perturb timing; audit is not a benchmark; no user data/checkpoints",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
