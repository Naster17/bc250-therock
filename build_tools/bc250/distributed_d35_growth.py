#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Separate bounded ALL-gradient qualification and fresh D35 training measurement."""
import argparse
from datetime import timedelta
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import fixture_d35 as fixture
from capacity_plan_d35 import CONFIG, PARAMETERS, GIB, admit
from distributed_d32_capacity import meminfo
from distributed_d33_fresh import (
    Monitor,
    sample_memory,
    package_identity,
    forbidden_audit,
)
from distributed_d05_train import atomic_json, sha

from artifact_d35 import (
    source_hashes as sources,
    load_manifest,
    read_blob,
    write_blob,
    admit_storage,
)
import shutil


def load_proof(path, wanted, mode, rank):
    if not wanted or sha(path) != wanted:
        raise ValueError("Qualification proof digest changed")
    proof = json.loads(path.read_text())
    if not (
        proof["status"] == "passed"
        and proof["phase"] == "qualify"
        and proof["mode"] == mode
        and proof["physical_rank"] == rank
        and proof["config"] == CONFIG
        and proof["sources"] == sources()
        and proof["reference"]["gradient_elements"] == PARAMETERS
        and proof["reference"]["ALL_AdamW_parameters_moments_counters"] is True
    ):
        raise ValueError("Shape/source/rank qualification requires review")
    return proof


def tensor_blob(directory, item, shape, dtype):
    return torch.frombuffer(read_blob(directory, item), dtype=dtype).reshape(shape)


def qualify(model, opt, rank, world, traffic, monitor, manifest_path, manifest):
    monitor.phase = "GPU-backward-and-staged-CPU-reference"
    directory = manifest_path.parent
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    actual_directory = manifest_path.parent.parent / (
        "actual-" + ("single" if world == 1 else "fsdp")
    )
    actual_directory.mkdir()
    scratch = {}
    opt.zero_grad(set_to_none=True)
    actual_loss = fixture.gpu_backward(model, 0, rank, world, traffic)
    if world == 2:
        dist.all_reduce(actual_loss)
        actual_loss /= world
    torch.testing.assert_close(
        actual_loss.cpu().double(),
        torch.tensor(manifest["CPU_FP64_loss"], dtype=torch.float64),
        atol=3e-6,
        rtol=5e-5,
    )
    squares = 0.0
    max_error = 0.0
    with fixture.full(model, True):
        current = fixture.named(model)
        assert set(current) == set(entries)
        root = model.module if isinstance(model, FSDP) else model
        assert root.get_input_embeddings().weight is root.get_output_embeddings().weight
        for index, (name, p) in enumerate(current.items()):
            entry = entries[name]
            assert p.is_cuda and p.dtype == p.grad.dtype == torch.float32
            initial = tensor_blob(
                directory, entry["initial"], entry["shape"], torch.float32
            )
            assert torch.equal(p.detach().cpu(), initial)
            expected = tensor_blob(
                directory, entry["gradient"], entry["shape"], torch.float64
            )
            raw = p.grad.detach().cpu().contiguous()
            actual = raw.double()
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-4)
            max_error = max(max_error, float((actual - expected).abs().max()))
            squares += float(actual.square().sum())
            scratch[name] = write_blob(
                actual_directory, f"{index:03d}-actual.raw", memoryview(raw.numpy())
            )
            del initial, expected, raw, actual
        del current, p
    wanted_norm = math.sqrt(squares)
    norm = clip(model)
    torch.testing.assert_close(
        norm.cpu().double(),
        torch.tensor(wanted_norm, dtype=torch.float64),
        atol=1e-7,
        rtol=1e-5,
    )
    coefficient = min(1.0, 0.5 / (wanted_norm + 1e-6))
    with fixture.full(model, True):
        for name, p in fixture.named(model).items():
            raw = tensor_blob(
                actual_directory, scratch[name], entries[name]["shape"], torch.float32
            ).double()
            torch.testing.assert_close(
                p.grad.detach().cpu().double(), raw * coefficient, atol=1e-8, rtol=5e-5
            )
            del raw
            # These are newly-created per-run audit scratch, not reference/evidence/checkpoint files.
            (actual_directory / scratch[name]["file"]).unlink()
        del p
    actual_directory.rmdir()
    monitor.phase = "GPU-AdamW-and-staged-per-parameter-CPU-checks"
    opt.step()
    maximum_CPU_moment_elements = 0
    with fixture.full(model, True):
        for name, p in fixture.named(model).items():
            entry = entries[name]
            initial = tensor_blob(
                directory, entry["initial"], entry["shape"], torch.float32
            )
            ref = torch.nn.Parameter(initial.double())
            ref.grad = p.grad.detach().cpu().double()
            refopt = fixture.optimizer(torch.nn.ParameterList([ref]))
            refopt.step()
            maximum_CPU_moment_elements = max(
                maximum_CPU_moment_elements, 2 * ref.numel()
            )
            torch.testing.assert_close(
                p.detach().cpu().double(), ref.detach(), atol=4e-6, rtol=5e-5
            )
            del initial, refopt, ref
        del p
    local_elements = 0
    for p in model.parameters():
        if p.numel() == 0:
            continue
        assert p.grad is not None
        state = opt.state[p]
        grad = p.grad.detach().cpu().double()
        for key, expected_moment, atol, rtol in [
            ("exp_avg", grad * 0.1, 3e-7, 5e-4),
            ("exp_avg_sq", grad.square() * 0.001, 3e-9, 1e-3),
        ]:
            torch.testing.assert_close(
                state[key].cpu().double(), expected_moment, atol=atol, rtol=rtol
            )
        assert int(state["step"]) == 1
        local_elements += p.numel()
    assert local_elements == PARAMETERS // world
    return dict(
        loss=float(actual_loss),
        CPU_FP64_loss=manifest["CPU_FP64_loss"],
        gradient_parameters=len(entries),
        gradient_elements=PARAMETERS,
        maximum_absolute_gradient_error=max_error,
        root_norm=float(norm),
        CPU_root_norm=wanted_norm,
        clip_coefficient=coefficient,
        ALL_AdamW_parameters_moments_counters=True,
        local_moment_elements_each=local_elements,
        maximum_CPU_moment_elements=maximum_CPU_moment_elements,
        actual_raw_spool_bytes=PARAMETERS * 4,
        actual_raw_spool_removed=True,
        actual_raw_spool_sha256={
            name: item["sha256"] for name, item in scratch.items()
        },
        full_CPU_optimizer_or_export=False,
        CPU_oracle_executed_in_GPU_worker=False,
    )


def clip(model):
    return (
        model.clip_grad_norm_(0.5)
        if isinstance(model, FSDP)
        else torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "fsdp"], required=True)
    p.add_argument("--phase", choices=["qualify", "measure"], required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-sha256", required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--qualification-sha256")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(status="running", mode=a.mode, phase=a.phase, config=CONFIG, checks=[])
    monitor = None
    start_worker = time.perf_counter()
    try:
        physical_rank = int(os.environ["RANK"])
        package_identity()
        manifest = load_manifest(a.reference, a.reference_sha256, CONFIG, PARAMETERS)
        assert manifest["torch"] == torch.__version__
        r["CPU_artifact_manifest_sha256"] = a.reference_sha256
        r["CPU_oracle_executed_in_GPU_worker"] = False
        if a.phase == "qualify":
            r["storage"] = admit_storage(
                shutil.disk_usage(a.output.parent).free, PARAMETERS, producing=False
            )
        proof = None
        if a.phase == "measure":
            proof = load_proof(
                a.qualification, a.qualification_sha256, a.mode, physical_rank
            )
            assert proof["torch"] == torch.__version__
            r["qualification_sha256"] = a.qualification_sha256
            r["qualification_birth_uuid"] = proof["birth"]["uuid"]
            FSDP.full_optim_state_dict = forbidden_audit
            FSDP.summon_full_params = forbidden_audit
        else:
            FSDP.full_optim_state_dict = forbidden_audit
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert len(os.sched_getaffinity(0)) <= 4
        world = 1 if a.mode == "single" else 2
        if world == 2:
            dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        rank = dist.get_rank() if world == 2 else 0
        r.update(
            rank=rank,
            physical_rank=physical_rank,
            world=world,
            torch=torch.__version__,
            sources=sources(),
            plan=admit(a.mode, meminfo()["MemAvailable"], torch.cuda.mem_get_info()[0]),
            birth=dict(
                uuid=str(uuid.uuid4()),
                pid=os.getpid(),
                start_ticks=Path("/proc/self/stat")
                .read_text()
                .split(") ", 1)[1]
                .split()[19],
            ),
            CPU_reference_executed_in_this_worker=False,
            full_CPU_optimizer_or_export=False,
        )
        monitor = Monitor()
        before = sample_memory()["events"]
        r["cgroup_events_before"] = before
        snapshots = []

        def snapshot(stage):
            torch.cuda.synchronize()
            snapshots.append(
                dict(
                    stage=stage,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    **sample_memory(),
                )
            )

        snapshot("before-model")
        model = fixture.build().cuda()
        assert sum(x.numel() for x in model.parameters()) == PARAMETERS
        if a.mode == "fsdp":
            model = FSDP(
                model,
                device_id=0,
                use_orig_params=True,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block}
                ),
                limit_all_gathers=True,
            )
        opt = fixture.optimizer(model)
        snapshot("after-model-wrap")
        traffic = dict(active=False, gathers=[], reductions=[])
        native_gather, native_reduce = (
            dist.all_gather_single,
            dist.reduce_scatter_single,
        )

        def gather(output, input, *args, **kwargs):
            if traffic["active"]:
                assert input.dtype == output.dtype == torch.float32
                traffic["gathers"].append(input.numel() * input.element_size())
            return native_gather(output, input, *args, **kwargs)

        def reduce(output, input, *args, **kwargs):
            if traffic["active"]:
                assert input.dtype == output.dtype == torch.float32
                traffic["reductions"].append(input.numel() * input.element_size())
            return native_reduce(output, input, *args, **kwargs)

        dist.all_gather_single, dist.reduce_scatter_single = gather, reduce
        if a.phase == "qualify":
            r["reference"] = qualify(
                model, opt, rank, world, traffic, monitor, a.reference, manifest
            )
            r["checks"] = [
                "identical-initialization-tied-head",
                "staged-CPU-FP64-loss-ALL-gradients",
                "CPU-root-norm-disk-spooled-ALL-clipped-gradients",
                "streamed-CPU-AdamW-ALL-parameters",
                "ALL-local-AdamW-moments-counters",
                "no-full-CPU-optimizer-or-export",
            ]
            snapshot("after-qualification")
        else:

            def step(index):
                opt.zero_grad(set_to_none=True)
                value = fixture.gpu_backward(model, index, rank, world, traffic)
                clip(model)
                opt.step()
                return value

            monitor.phase = "initial-GPU-update"
            first = step(0)
            if world == 2:
                dist.all_reduce(first)
                first /= world
            r["initial_global_loss"] = float(first)
            assert abs(float(first) - proof["reference"]["loss"]) <= 1e-5
            snapshot("after-initial-update")
            gc.collect()
            torch.cuda.empty_cache()
            monitor.phase = "warmup"
            for index in [1, 2]:
                step(index)
            snapshot("after-warmup")
            monitor.phase = "measured-training"
            samples = []
            for index in range(3, 8):
                if world == 2:
                    dist.barrier()
                torch.cuda.synchronize()
                previous = {k: len(traffic[k]) for k in ["gathers", "reductions"]}
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                value = step(index)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                metrics = dict(
                    seconds_local=seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    **sample_memory(),
                )
                control = torch.tensor(
                    [seconds, float(value)], dtype=torch.float64, device="cuda"
                )
                if world == 2:
                    dist.all_reduce(control[:1], op=dist.ReduceOp.MAX)
                    dist.all_reduce(control[1:])
                metrics.update(
                    seconds_max_rank=float(control[0]),
                    global_loss=float(control[1]) / world,
                    traffic_input_bytes={
                        k: sum(traffic[k][previous[k] :]) for k in previous
                    },
                )
                assert (
                    math.isfinite(metrics["global_loss"])
                    and metrics["host_available_bytes"] > 3 * GIB
                )
                samples.append(metrics)
            r["samples"] = samples
            r["median_update_seconds"] = statistics.median(
                s["seconds_max_rank"] for s in samples
            )
            r["checks"] = [
                "source-shape-bound-qualification",
                "fresh-no-CPU-or-full-audits",
                "initial-loss-parity",
                "native-FP32-training-collectives",
                "five-finite-updates",
            ]
            snapshot("after-measured-training")
        local_elements = sum(x.numel() for x in model.parameters())
        state_elements = sum(s["exp_avg"].numel() for s in opt.state.values())
        assert local_elements == state_elements == PARAMETERS // world
        r["state"] = dict(
            full_parameter_elements=PARAMETERS,
            local_master_elements=local_elements,
            AdamW_moment_elements_each=state_elements,
            AdamW_state_bytes=sum(
                v.numel() * v.element_size()
                for s in opt.state.values()
                for v in s.values()
                if isinstance(v, torch.Tensor)
            ),
        )
        r["snapshots"] = snapshots
        if world == 2:
            dist.barrier()
        after = sample_memory()["events"]
        r["cgroup_events_after"] = after
        assert all(before[k] == after[k] == 0 for k in ["oom", "oom_kill"])
        r.update(status="passed", worker_seconds=time.perf_counter() - start_worker)
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        if monitor is not None:
            r["memory_samples"] = monitor.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
        atomic_json(a.output, r)


if __name__ == "__main__":
    main()
