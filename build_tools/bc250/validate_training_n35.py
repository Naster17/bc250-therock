#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual-geometry accumulation16 and owned optimizer-boundary checkpoint replay."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from profile_training_n31 import cpu_tree, identical
import validate_qwen35_training_model as existing


ACCUMULATION = 16


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rng():
    return dict(
        cpu=torch.get_rng_state(),
        gpu=torch.cuda.get_rng_state(),
        python=random.getstate(),
    )


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["gpu"])
    random.setstate(state["python"])


def schedule(step):
    return min((step + 1) / 2, 1.0) * (1 + math.cos(math.pi * min(step, 4) / 4)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--checkpoint-test-sha",
        help="Explicit origin qualifier SHA for an owned retained checkpoint",
    )
    parser.add_argument("--state-name", default="resumed-step2.pt")
    args = parser.parse_args()
    assert args.scratch_root.is_dir() and args.scratch_root.parent == Path("/tmp")
    preset = Path("/workspace/rocblas-gemm-overrides.csv")
    assert os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH") == str(preset)
    assert "LD_PRELOAD" not in os.environ
    torch.set_num_threads(4)
    torch.manual_seed(31935)
    random.seed(31935)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    app_path = Path("/home/nik/finetune-n7/replica/bc250_ops.py")
    spec = importlib.util.spec_from_file_location("n35_application", app_path)
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
    optimizer = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    identity = dict(
        preset_sha256=sha(preset),
        app_sha256=sha(app_path),
        model_helper_sha256=sha(Path(existing.__file__)),
        test_sha256=sha(Path(__file__)),
        native_torch=str(torch.__version__),
        hip=str(torch.version.hip),
        architecture="gfx1013",
        length=1024,
        accumulation=16,
        parameter_names=list(params),
    )
    actual_test_sha = identity["test_sha256"]
    if args.checkpoint_test_sha:
        assert args.resume and len(args.checkpoint_test_sha) == 64
        identity["test_sha256"] = args.checkpoint_test_sha
    assert Path(args.state_name).name == args.state_name and args.state_name.endswith(
        ".pt"
    )
    report = dict(
        actual_test_sha256=actual_test_sha,
        status="running",
        checks=[],
        role="resumed" if args.resume else "uninterrupted",
        identity=identity,
        user_data_or_checkpoint_opened=False,
        optimizer_boundary_only=True,
        steps=[],
    )
    first_checkpoint = args.scratch_root / "checkpoint-step1.pt"

    def micro(label, index):
        begin = time.monotonic()
        nonces = [
            random.random(),
            float(torch.rand(())),
            float(torch.rand((), device="cuda")),
        ]
        ids = torch.randint(3, model.config.vocab_size, (1, 1024), device="cuda")
        token_sha = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=ids, labels=ids, use_cache=False)
            loss = output.loss
            del output
        (loss / ACCUMULATION).backward()
        value = float(loss.detach())
        assert math.isfinite(value)
        print(
            f"N35 {label} {index + 1}/16 loss={value:.9f} seconds={time.monotonic() - begin:.3f}",
            flush=True,
        )
        return dict(loss=value, token_sha256=token_sha, nonces=nonces)

    def snapshot(step, rows, gradients=False):
        state = dict(
            identity=identity,
            parameters=cpu_tree(params),
            optimizer=cpu_tree(optimizer.state_dict()),
            scheduler=cpu_tree(scheduler.state_dict()),
            rng=cpu_tree(rng()),
            completed_steps=step,
            microbatches_seen=step * ACCUMULATION,
            rows=rows,
        )
        if gradients:
            state["gradients"] = {
                n: p.grad.detach().cpu().clone() for n, p in params.items()
            }
        return state

    def setup_cpu(parameters, opt_state=None, scheduler_state=None):
        ps = [torch.nn.Parameter(parameters[n].clone()) for n in params]
        opt = torch.optim.AdamW(ps, lr=1e-4, weight_decay=0.0, foreach=False)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        if opt_state is not None:
            # Optimizer loading may retain CPU step tensors. The CPU oracle
            # must own its state independently of the GPU optimizer.
            opt.load_state_dict(cpu_tree(opt_state))
            sched.load_state_dict(scheduler_state)
            for gp, cp in zip(params.values(), ps):
                gpu_counter = optimizer.state[gp]["step"]
                cpu_counter = opt.state[cp]["step"]
                if gpu_counter.device == cpu_counter.device:
                    assert gpu_counter.data_ptr() != cpu_counter.data_ptr()
        return ps, opt, sched

    def update(cpu_params, cpu_optimizer, cpu_scheduler):
        before = {n: p.grad.detach().cpu().clone() for n, p in params.items()}
        norm_cpu = math.sqrt(
            sum(float(g.double().square().sum()) for g in before.values())
        )
        norm_gpu = float(
            torch.nn.utils.clip_grad_norm_(
                params.values(), 1.0, error_if_nonfinite=True
            )
        )
        assert abs(norm_cpu - norm_gpu) <= 1e-6 + 1e-6 * norm_cpu
        factor = min(1.0, 1.0 / (norm_cpu + 1e-6))
        for (n, p), cp in zip(params.items(), cpu_params):
            actual = p.grad.detach().cpu()
            torch.testing.assert_close(
                actual.double(), before[n].double() * factor, atol=1e-8, rtol=1e-5
            )
            cp.grad = actual.clone()
        optimizer.step()
        scheduler.step()
        cpu_optimizer.step()
        cpu_scheduler.step()
        error = 0.0
        for (n, p), cp in zip(params.items(), cpu_params):
            actual = p.detach().cpu()
            torch.testing.assert_close(actual, cp, atol=2e-7, rtol=1e-5)
            error = max(error, float((actual - cp.detach()).abs().max()))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer.state[p][key].cpu(),
                    cpu_optimizer.state[cp][key],
                    atol=1e-12,
                    rtol=1e-5,
                )
        identical(scheduler.state_dict(), cpu_scheduler.state_dict())
        return dict(
            gradient_norm_cpu=norm_cpu,
            gradient_norm_gpu=norm_gpu,
            cpu_adamw_parameter_max_abs_error=error,
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    try:
        if args.resume:
            state = torch.load(args.resume, weights_only=True, map_location="cpu")
            assert (
                state["identity"] == identity
                and state["completed_steps"] == 1
                and state["microbatches_seen"] == 16
            )
            assert state["parameters"].keys() == params.keys()
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(state["parameters"][n])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            cpu_params, cpu_optimizer, cpu_scheduler = setup_cpu(
                state["parameters"], state["optimizer"], state["scheduler"]
            )
            restore_rng(state["rng"])
            optimizer.zero_grad(set_to_none=True)
            rows = [micro("fresh-resume", i) for i in range(16)]
            report["steps"].append(update(cpu_params, cpu_optimizer, cpu_scheduler))
            final = args.scratch_root / args.state_name
            assert not final.exists()
            torch.save(snapshot(2, rows, gradients=True), final)
            report.update(
                status="passed", state_file=final.name, state_sha256=sha(final)
            )
        else:
            initial = cpu_tree(params)
            cpu_params, cpu_optimizer, cpu_scheduler = setup_cpu(initial)
            start_rng = cpu_tree(rng())
            summed = {
                n: torch.zeros_like(p, device="cpu", dtype=torch.float64)
                for n, p in params.items()
            }
            reference_rows = []
            for i in range(16):
                optimizer.zero_grad(set_to_none=True)
                reference_rows.append(micro("independent-reference", i))
                for n, p in params.items():
                    assert p.grad is not None
                    summed[n].add_(p.grad.detach().cpu().double())
            end_rng = cpu_tree(rng())
            assert not optimizer.state
            identical(cpu_tree(params), initial)
            reference_path = args.scratch_root / "independent-fp64-sum.pt"
            torch.save(summed, reference_path)
            restore_rng(start_rng)
            optimizer.zero_grad(set_to_none=True)
            actual_rows = [micro("accumulation16", i) for i in range(16)]
            assert actual_rows == reference_rows
            identical(cpu_tree(rng()), end_rng)
            errors = {}
            for n, p in params.items():
                actual = p.grad.detach().cpu().double()
                expected = summed[n]
                assert bool(actual.isfinite().all()) and float(expected.norm()) > 0
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-5)
                errors[n] = float((actual - expected).norm() / expected.norm())
            assert max(errors.values()) < 2e-6
            report.update(
                max_accumulation_gradient_relative_l2=max(errors.values()),
                accumulation_gradient_count=len(errors),
                reference_gradient_sha256=sha(reference_path),
            )
            report["checks"] += [
                "fp64-sum16-372-gradients",
                "exact-loss-inputs-three-rng",
            ]
            del summed, initial
            report["steps"].append(update(cpu_params, cpu_optimizer, cpu_scheduler))
            report["checks"] += [
                "independent-global-clip",
                "first-cpu-adamw-scheduler-state",
            ]
            torch.save(snapshot(1, actual_rows), first_checkpoint)
            report["checks"].append("owned-checkpoint-identity-progress")
            optimizer.zero_grad(set_to_none=True)
            second_rows = [micro("uninterrupted-step2", i) for i in range(16)]
            report["steps"].append(update(cpu_params, cpu_optimizer, cpu_scheduler))
            report["checks"].append("second-cpu-adamw-scheduler-state")
            uninterrupted = args.scratch_root / "uninterrupted-step2.pt"
            torch.save(snapshot(2, second_rows, gradients=True), uninterrupted)
            # Release GPU state before the fresh process loads the same immutable base.
            del (
                cpu_params,
                cpu_optimizer,
                cpu_scheduler,
                params,
                optimizer,
                scheduler,
                model,
            )
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            resumed_json = args.output.parent / "resumed-worker.json"
            command = [
                sys.executable,
                str(Path(__file__)),
                "--output",
                str(resumed_json),
                "--scratch-root",
                str(args.scratch_root),
                "--resume",
                str(first_checkpoint),
            ]
            with (args.output.parent / "resumed-worker.log").open("w") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
            resumed = json.loads(resumed_json.read_text())
            assert resumed["status"] == "passed"
            a = torch.load(uninterrupted, weights_only=True, map_location="cpu")
            b = torch.load(
                args.scratch_root / resumed["state_file"],
                weights_only=True,
                map_location="cpu",
            )
            identical(a, b)
            report["checks"] += [
                "fresh-process-exact-372-parameters-gradients-adamw-scheduler",
                "fresh-process-exact-loss-inputs-progress-three-rng",
            ]
            report.update(
                status="passed",
                resumed_worker=resumed,
                artifacts={
                    p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                    for p in sorted(args.scratch_root.glob("*.pt"))
                },
            )
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
