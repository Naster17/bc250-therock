#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fresh-process AB/BA whole-model training checks for one rocBLAS override."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import validate_qwen35_training_model as existing


def worker(args, *, app_path=None):
    torch.manual_seed(31932)
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        "n32_model_app", app_path or "/home/nik/finetune-n7/replica/bc250_ops.py"
    )
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
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(params) == 372 and all(p.dtype == torch.float32 for p in params.values())
    ids = torch.randint(3, model.config.vocab_size, (1, 1024), device="cuda")
    optimizer = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=0.0)
    before = {n: p.detach().cpu().clone() for n, p in params.items()}

    def step():
        optimizer.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach()

    loss = step()
    torch.cuda.synchronize()
    gradients = {n: p.grad.detach().cpu().clone() for n, p in params.items()}
    assert all(
        bool(g.isfinite().all()) and float(g.norm()) > 0 for g in gradients.values()
    )
    # Independent CPU AdamW verifies this worker's own clipped-gradient update.
    cpu_params = [torch.nn.Parameter(before[n]) for n in params]
    cpu_optimizer = torch.optim.AdamW(
        cpu_params, lr=1e-4, weight_decay=0.0, foreach=False
    )
    for p, g in zip(cpu_params, gradients.values()):
        p.grad = g
    cpu_optimizer.step()
    max_update_error = 0.0
    for index, ((name, p), expected) in enumerate(zip(params.items(), cpu_params)):
        actual = p.detach().cpu()
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)
        max_update_error = max(
            max_update_error, float((actual - expected.detach()).abs().max())
        )
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[p][key].cpu(),
                cpu_optimizer.state[expected][key],
                atol=1e-12,
                rtol=1e-5,
            )
    grad_path = args.output.with_suffix(".gradients.pt")
    torch.save(gradients, grad_path)
    result = dict(
        status="running",
        mode=args.worker,
        loss=float(loss),
        gradient_count=len(gradients),
        cpu_adamw_updates_checked=len(params),
        max_cpu_adamw_update_abs_error=max_update_error,
        gradients=grad_path.name,
        gradient_sha256=hashlib.sha256(grad_path.read_bytes()).hexdigest(),
        cpu_rng_sha256=hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest(),
        gpu_rng_sha256=hashlib.sha256(
            torch.cuda.get_rng_state().numpy().tobytes()
        ).hexdigest(),
        samples=[],
        override=os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH"),
    )
    del gradients, before, cpu_params, cpu_optimizer
    gc.collect()
    for _ in range(3):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        loss = step()
        end.record()
        end.synchronize()
        result["samples"].append(
            dict(
                wall_seconds=time.perf_counter() - begin,
                gpu_event_ms=start.elapsed_time(end),
                loss=float(loss),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
        )
    result.update(
        status="passed",
        median_wall_seconds=statistics.median(
            s["wall_seconds"] for s in result["samples"]
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("reference", "candidate"))
    parser.add_argument("--override", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    assert args.override and args.override.is_file()
    report = dict(status="running", checks=[], results=[])
    reference_gradients = None
    reference = None
    try:
        for index, mode in enumerate(
            ("reference", "candidate", "candidate", "reference")
        ):
            output = args.output.parent / f"worker-{index}-{mode}.json"
            env = dict(os.environ)
            env.pop("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH", None)
            if mode == "candidate":
                env["ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH"] = str(args.override)
            command = [
                sys.executable,
                str(Path(__file__)),
                "--worker",
                mode,
                "--output",
                str(output),
            ]
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    env=env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=150,
                )
            result = json.loads(output.read_text())
            assert result["status"] == "passed" and result["gradient_count"] == 372
            gradients = torch.load(
                output.with_suffix(".gradients.pt"),
                weights_only=True,
                map_location="cpu",
            )
            if reference is None:
                reference = result
                reference_gradients = gradients
            else:
                assert result["loss"] == reference["loss"]
                assert gradients.keys() == reference_gradients.keys()
                errors = {
                    n: float(
                        (g - reference_gradients[n]).norm()
                        / reference_gradients[n].norm().clamp_min(1e-12)
                    )
                    for n, g in gradients.items()
                }
                assert max(errors.values()) < 0.04, (mode, max(errors.values()))
                result["max_adapter_gradient_relative_l2"] = max(errors.values())
                result["exact_gradients"] = all(
                    torch.equal(gradients[n], reference_gradients[n]) for n in gradients
                )
                assert result["cpu_rng_sha256"] == reference["cpu_rng_sha256"]
                assert result["gpu_rng_sha256"] == reference["gpu_rng_sha256"]
            report["results"].append(result)
            report["checks"].append(f"full-model-{index}-{mode}-gradients-adamw-rng")
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["paired_wall_ratios"] = [
            report["results"][0]["median_wall_seconds"]
            / report["results"][1]["median_wall_seconds"],
            report["results"][3]["median_wall_seconds"]
            / report["results"][2]["median_wall_seconds"],
        ]
        assert min(report["paired_wall_ratios"]) > 1.1, report["paired_wall_ratios"]
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
