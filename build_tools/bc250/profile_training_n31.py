#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded synthetic Qwen3.5 LoRA step profile with exact unprofiled replay."""
import argparse
import contextlib
import functools
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile, record_function

import validate_qwen35_training_model as existing
from analyze_training_n31 import analyze

PROJECT = Path("/home/nik/finetune-n7")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def identical(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            identical(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            identical(a, b)
    else:
        assert actual == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(65, 1024), default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(31919)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    spec = importlib.util.spec_from_file_location(
        "n31_application_ops", PROJECT / "replica/bc250_ops.py"
    )
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    model = existing.make_model(True, torch.bfloat16).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.001)
    app.install(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    assert model.training and model.is_gradient_checkpointing
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(parameters) == 372
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4, weight_decay=0.0)
    ids = torch.randint(3, model.config.vocab_size, (1, args.length), device="cuda")
    capabilities = dict(
        distributed=torch.distributed.is_available(),
        flash_attention=torch.backends.cuda.is_flash_attention_available(),
        torch_hip_version=torch.version.hip,
        optional_packages={},
    )
    for package in (
        "torchvision",
        "torchaudio",
        "torchao",
        "bitsandbytes",
        "onnx",
        "onnxruntime",
        "flash-linear-attention",
        "causal-conv1d",
    ):
        try:
            capabilities["optional_packages"][package] = importlib.metadata.version(
                package
            )
        except importlib.metadata.PackageNotFoundError:
            capabilities["optional_packages"][package] = None
    profiling = False
    stage_calls = {}
    patched = []

    def wrap(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*values, **kwargs):
            if profiling:
                stage_calls[label] = stage_calls.get(label, 0) + 1
            with record_function(label) if profiling else contextlib.nullcontext():
                return original(*values, **kwargs)

        setattr(owner, name, measured)
        patched.append((owner, name, original))

    wrap(existing.qwen, "torch_chunk_gated_delta_rule", "n31/gated_delta")
    wrap(existing.qwen, "causal_conv1d_fn", "n31/causal_conv")
    wrap(model.get_base_model(), "loss_function", "n31/chunked_loss")

    def stage(name):
        return record_function("n31/" + name) if profiling else contextlib.nullcontext()

    def step():
        with stage("zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            with stage("forward"), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=ids, labels=ids, use_cache=False)
                loss = output.loss
                del output
            with stage("backward"):
                loss.backward()
        with stage("clip"):
            torch.nn.utils.clip_grad_norm_(
                parameters.values(), 1.0, error_if_nonfinite=True
            )
        with stage("optimizer"):
            optimizer.step()
        return loss.detach()

    def capture():
        return dict(
            parameters=cpu_tree(parameters),
            gradients={n: cpu_tree(p.grad) for n, p in parameters.items()},
            optimizer=cpu_tree(optimizer.state_dict()),
            cpu_rng=torch.get_rng_state().clone(),
            gpu_rng=torch.cuda.get_rng_state().clone(),
        )

    result = dict(
        status="running",
        capabilities=capabilities,
        checks=[],
        length=args.length,
        batch=1,
        rank=16,
        dtype="bfloat16 base/autocast with FP32 adapters",
        attention="math",
        accumulation_steps=1,
        gradient_checkpointing=True,
        synthetic_inputs=True,
        user_data_or_checkpoints_opened=False,
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(existing.__file__), Path(app.__file__))
        },
        protocol="warmup1; exact unprofiled/profiled loss, clipped gradients, parameters, optimizer and RNG replay; three warm unprofiled whole steps; no throughput improvement claim",
    )
    try:
        print("N31 model ready; warming full step", flush=True)
        step()
        torch.cuda.synchronize()
        before = capture()
        reference_loss = step()
        torch.cuda.synchronize()
        reference_value = float(reference_loss)
        reference = capture()
        del reference_loss
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(before["parameters"][name])
        optimizer.load_state_dict(before["optimizer"])
        torch.set_rng_state(before["cpu_rng"])
        torch.cuda.set_rng_state(before["gpu_rng"])
        del before
        torch.cuda.synchronize()
        profiling = True
        begin = time.perf_counter()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            with record_function("n31/training_step"):
                actual_loss = step()
                torch.cuda.synchronize()
        result["profiled_wall_seconds_with_setup_teardown"] = (
            time.perf_counter() - begin
        )
        profiling = False
        actual = capture()
        assert float(actual_loss) == reference_value and math.isfinite(reference_value)
        identical(actual, reference)
        assert all(
            g is not None and bool(torch.isfinite(g).all()) and float(g.norm()) > 0
            for g in actual["gradients"].values()
        )
        result["checks"] += [
            "exact-loss",
            "372-exact-nonzero-gradients",
            "exact-adapter-update",
            "exact-optimizer-state",
            "exact-cpu-gpu-rng",
        ]
        result["loss"] = reference_value
        del reference, actual, actual_loss
        trace = args.output.parent / "training-step-trace.json"
        prof.export_chrome_trace(str(trace))
        assert trace.stat().st_size < 256 * 1024**2
        # Parse kernel records rather than summing CPU ops, kernels and GPU
        # annotation views of the same work. Capture metadata includes mm shapes.
        result["kernel_analysis"] = analyze(
            json.loads(trace.read_text())["traceEvents"]
        )
        result["stage_calls"] = stage_calls
        assert stage_calls["n31/gated_delta"] == 36
        assert stage_calls["n31/causal_conv"] == 36
        assert stage_calls["n31/chunked_loss"] == 1
        result.update(
            gpu_kernel_count=result["kernel_analysis"]["kernel_count"],
            trace_bytes=trace.stat().st_size,
            trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        )
        result["checks"] += ["positive-native-gpu-events", "checkpointed-stage-calls"]
        samples = []
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
            samples.append(
                dict(
                    wall_seconds=time.perf_counter() - begin,
                    gpu_event_ms=start.elapsed_time(end),
                    loss=float(loss),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                )
            )
        assert all(math.isfinite(s["loss"]) and s["gpu_event_ms"] > 0 for s in samples)
        result.update(
            unprofiled_samples=samples,
            median_unprofiled_wall_seconds=statistics.median(
                s["wall_seconds"] for s in samples
            ),
            status="passed",
        )
        result["checks"].append("three-finite-warm-training-steps")
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("N31_PROFILE_PASS", len(result["checks"]), flush=True)


if __name__ == "__main__":
    main()
