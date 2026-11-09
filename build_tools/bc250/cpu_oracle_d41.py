#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only checkpointed FP64 oracle with per-gradient publication and roundtrips."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import time
import uuid
import torch
import torch.utils.checkpoint as checkpoint_module
from artifact_d35 import load_manifest, read_blob, write_blob, digest
from artifact_d41 import sources, validate
from capacity_plan_d41 import admit, headroom
from cpu_checkpoint_policy_d38 import OPTIONS, validate_options, callable_matches
from fixture_d41 import select
from distributed_d32_capacity import meminfo
from distributed_d33_fresh import Monitor, sample_memory, package_identity
from distributed_d05_train import atomic_json
from stream_contract_d37 import Completion

CHECKPOINT_SHA = "eab49639231cc989d65c04212923eef0ade4aa059b68bcabfe2819007f34e69f"


def tensor_sha(tensor):
    value = tensor.detach().contiguous()
    assert value.device.type == "cpu"
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["control", "large"], required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = select(args.shape)
    config = fixture.CONFIG
    result = dict(
        status="running", shape=args.shape, config=config, sources=sources(), checks=[]
    )
    monitor, handles = None, []
    start = time.perf_counter()
    snapshots = []
    try:
        package_identity()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        assert not torch.cuda.is_initialized() and len(os.sched_getaffinity(0)) <= 4
        plan = admit(
            args.shape,
            meminfo()["MemAvailable"],
            shutil.disk_usage(args.output.parent).free,
        )
        expected = None
        if args.shape == "control":
            expected = load_manifest(
                args.reference, args.reference_sha256, config, plan["parameters"]
            )
            assert expected["torch"] == torch.__version__
        else:
            assert args.reference is args.reference_sha256 is None
        directory = args.output.parent / (
            "reference" if args.shape == "large" else "roundtrip-scratch"
        )
        directory.mkdir()
        result.update(
            plan=plan,
            parameters=plan["parameters"],
            torch=torch.__version__,
            physical_rank=0,
            CUDA_initialized=False,
            checkpoint_options=dict(OPTIONS),
            checkpoint_source_sha256=digest(Path(checkpoint_module.__file__)),
            birth=dict(uuid=str(uuid.uuid4()), pid=os.getpid()),
        )
        assert result["checkpoint_source_sha256"] == CHECKPOINT_SHA
        monitor = Monitor()
        before = sample_memory()["events"]
        result["cgroup_events_before"] = before

        def check(stage):
            sample = sample_memory()
            free = shutil.disk_usage(args.output.parent).free
            headroom(sample["memory.current"], sample["host_available_bytes"], free)
            snapshots.append(dict(stage=stage, free_disk_bytes=free, **sample))

        def gate():
            sample = sample_memory()
            headroom(
                sample["memory.current"],
                sample["host_available_bytes"],
                shutil.disk_usage(args.output.parent).free,
            )

        check("before-model")
        monitor.phase = "CPU-initialization"
        model = fixture.build().double()
        named = fixture.named(model)
        assert sum(p.numel() for p in named.values()) == plan["parameters"]
        assert len(named) == (148 if args.shape == "control" else 244)
        assert (
            model.get_input_embeddings().weight is model.get_output_embeddings().weight
        )
        assert not model.is_gradient_checkpointing
        validate_options(dict(OPTIONS))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(OPTIONS))
        modules = [m for m in model.modules() if hasattr(m, "gradient_checkpointing")]

        def checkpoint_runtime():
            assert (
                model.is_gradient_checkpointing and len(modules) == config["blocks"] + 1
            )
            assert all(
                m.gradient_checkpointing is True
                and callable_matches(
                    m._gradient_checkpointing_func, checkpoint_module.checkpoint
                )
                for m in modules
            )

        checkpoint_runtime()
        check("after-model")
        expected_entries = (
            {e["name"]: e for e in expected["tensors"]} if expected else None
        )
        if expected_entries is not None:
            assert set(named) == set(expected_entries)
        entries = {}
        initial_hashes = {}

        def roundtrip(raw, filename):
            gate()
            item = write_blob(directory, filename, memoryview(raw.numpy()))
            restored = torch.frombuffer(
                read_blob(directory, item), dtype=raw.dtype
            ).reshape(raw.shape)
            assert torch.equal(restored, raw) and item["sha256"] == tensor_sha(raw)
            del restored
            if args.shape == "control":
                (directory / filename).unlink()
            return item

        for index, (name, p) in enumerate(named.items()):
            gate()
            assert p.device.type == "cpu" and p.dtype == torch.float64
            raw = p.detach().float().contiguous()
            assert torch.equal(p.detach(), raw.double())
            item = roundtrip(raw, f"{index:03d}-initial.raw")
            if expected_entries:
                assert item["sha256"] == expected_entries[name]["initial"]["sha256"]
            initial_hashes[name] = item["sha256"]
            entries[name] = dict(name=name, shape=list(p.shape), initial=item)
            del raw
        check("after-initial-publication")
        completion = Completion(entries)
        hook_counts = {n: 0 for n in entries}
        gradient_hashes = {}
        in_hook = False
        maximum_visible = maximum_roundtrip = 0

        def make_hook(name, index):
            def hook(p):
                nonlocal in_hook, maximum_visible, maximum_roundtrip
                assert not in_hook
                in_hook = True
                try:
                    gate()
                    completion.accept(name)
                    hook_counts[name] += 1
                    assert (
                        p.grad is not None
                        and p.grad.device.type == "cpu"
                        and p.grad.dtype == torch.float64
                    )
                    assert (
                        list(p.grad.shape) == entries[name]["shape"]
                        and torch.isfinite(p.grad).all()
                    )
                    maximum_visible = max(
                        maximum_visible,
                        sum(
                            q.grad.numel() for q in named.values() if q.grad is not None
                        ),
                    )
                    raw = p.grad.detach().contiguous()
                    item = roundtrip(raw, f"{index:03d}-gradient.raw")
                    maximum_roundtrip = max(maximum_roundtrip, item["bytes"])
                    if expected_entries:
                        old = expected_entries[name]["gradient"]
                        assert item["sha256"] == old["sha256"]
                        ref = torch.frombuffer(
                            read_blob(args.reference.parent, old), dtype=torch.float64
                        ).reshape(raw.shape)
                        assert torch.equal(raw, ref)
                        del ref
                    entries[name]["gradient"] = item
                    gradient_hashes[name] = item["sha256"]
                    del raw
                    p.grad = None
                finally:
                    in_hook = False

            return hook

        handles = [
            p.register_post_accumulate_grad_hook(make_hook(n, i))
            for i, (n, p) in enumerate(named.items())
        ]
        block_calls = {str(i): 0 for i in range(config["blocks"])}

        def count_forward(index):
            def hook(module, inputs):
                block_calls[str(index)] += 1

            return hook

        handles.extend(
            block.register_forward_pre_hook(count_forward(i))
            for i, block in enumerate(model.transformer.h)
        )
        checkpoint_runtime()
        rng_before = tensor_sha(torch.get_rng_state())
        monitor.phase = "CPU-forward"
        gate()
        ids, labels = fixture.data(0)
        value = fixture.loss(
            model(input_ids=ids).logits,
            labels,
            config["global_batch"] * (config["tokens"] - 1),
        )
        loss = float(value.detach())
        assert math.isfinite(loss)
        if expected:
            assert loss == expected["CPU_FP64_loss"]
        calls_forward = dict(block_calls)
        assert all(v == 1 for v in block_calls.values())
        check("after-forward")
        monitor.phase = "CPU-backward-stream-publication"
        value.backward()
        del value
        checkpoint_runtime()
        completion.finish()
        assert all(v == 2 for v in block_calls.values())
        assert all(v == 1 for v in hook_counts.values()) and all(
            p.grad is None for p in named.values()
        )
        assert maximum_visible <= plan["largest_FP64_tensor_bytes"] // 8
        assert maximum_roundtrip == plan["largest_FP64_tensor_bytes"]
        rng_after = tensor_sha(torch.get_rng_state())
        assert rng_before == rng_after
        check("after-backward")
        for handle in handles:
            handle.remove()
        handles = []
        monitor.phase = "CPU-initial-preservation-check"
        for name, p in named.items():
            gate()
            entry = entries[name] if args.shape == "large" else expected_entries[name]
            root = directory if args.shape == "large" else args.reference.parent
            initial = torch.frombuffer(
                read_blob(root, entry["initial"]), dtype=torch.float32
            ).reshape(p.shape)
            assert torch.equal(p.detach(), initial.double())
            del initial
        if args.shape == "control":
            assert not list(directory.iterdir())
            directory.rmdir()
        else:
            manifest = dict(
                schema=1,
                status="passed",
                config=config,
                parameters=plan["parameters"],
                sources=sources(),
                torch=torch.__version__,
                cuda_initialized=False,
                CPU_FP64_loss=loss,
                tensors=list(entries.values()),
            )
            atomic_json(directory / "manifest.json", manifest)
            result["manifest_sha256"] = digest(directory / "manifest.json")
            assert (
                validate(directory / "manifest.json", result["manifest_sha256"])
                == manifest
            )
        check("after-final-verification")
        assert not torch.cuda.is_initialized()
        after = sample_memory()["events"]
        assert all(before[k] == after[k] == 0 for k in ["oom", "oom_kill"])
        result.update(
            status="passed",
            CPU_FP64_loss=loss,
            gradient_parameters=len(entries),
            gradient_elements=plan["parameters"],
            initial_sha256=initial_hashes,
            gradient_sha256=gradient_hashes,
            hook_counts=hook_counts,
            retained_gradient_elements_after_backward=0,
            maximum_visible_gradient_elements_at_hook=maximum_visible,
            maximum_roundtrip_bytes=maximum_roundtrip,
            checkpoint_runtime_modules=len(modules),
            block_calls_after_forward=calls_forward,
            block_calls_after_backward=block_calls,
            CPU_rng_before_sha256=rng_before,
            CPU_rng_after_sha256=rng_after,
            snapshots=snapshots,
            cgroup_events_after=after,
            worker_seconds=time.perf_counter() - start,
            checks=[
                "CPU-only-source-config-parameter-admission",
                "FP64-initial-FP32-roundtrip-and-preservation",
                "ALL-finite-gradients-raw-roundtrip-SHA",
                "complete-once-hooks-zero-retained-gradients",
                "native-checkpoint-recipe-and-block-reentry",
                "dropout-zero-RNG-preserved",
                "bounded-RAM-disk-headroom-no-OOM",
                "complete-source-bound-oracle-or-exact-D35-control",
            ],
        )
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        for handle in handles:
            handle.remove()
        if monitor is not None:
            result["memory_samples"] = monitor.finish()
        result["snapshots"] = snapshots
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
