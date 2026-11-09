#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU FP64 full-gradient control versus exact post-accumulate gradient streaming."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import time
import uuid
import torch
import fixture_d35 as fixture
from capacity_plan_d35 import CONFIG, PARAMETERS, GIB
from artifact_d35 import source_hashes, load_manifest, read_blob, write_blob, digest
from distributed_d32_capacity import meminfo
from distributed_d33_fresh import Monitor, sample_memory, package_identity
from distributed_d05_train import atomic_json
from stream_contract_d37 import Completion, admit_scratch


def sources():
    result = source_hashes()
    result.update(
        {
            n: digest(Path(__file__).with_name(n))
            for n in ["cpu_stream_d37.py", "stream_contract_d37.py"]
        }
    )
    return result


def tensor_sha(tensor):
    value = tensor.detach().contiguous()
    assert value.device.type == "cpu"
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["retained", "stream"], required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        status="running",
        variant=args.variant,
        config=CONFIG,
        sources=sources(),
        checks=[],
    )
    monitor = None
    handles = []
    start = time.perf_counter()
    try:
        package_identity()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        assert not torch.cuda.is_initialized()
        assert (
            len(os.sched_getaffinity(0)) <= 4 and meminfo()["MemAvailable"] >= 12 * GIB
        )
        manifest = load_manifest(
            args.reference, args.reference_sha256, CONFIG, PARAMETERS
        )
        assert manifest["torch"] == torch.__version__
        entries = {e["name"]: e for e in manifest["tensors"]}
        bound = max(e["gradient"]["bytes"] for e in entries.values())
        result.update(
            torch=torch.__version__,
            physical_rank=int(os.environ["RANK"]),
            CPU_artifact_manifest_sha256=args.reference_sha256,
            storage=admit_scratch(shutil.disk_usage(args.output.parent).free, bound),
            birth=dict(
                uuid=str(uuid.uuid4()),
                pid=os.getpid(),
                start_ticks=Path("/proc/self/stat")
                .read_text()
                .split(") ", 1)[1]
                .split()[19],
            ),
            parameters=PARAMETERS,
            cuda_initialized=False,
            CPU_checkpointing_enabled=False,
        )
        monitor = Monitor()
        before = sample_memory()["events"]
        result["cgroup_events_before"] = before
        snapshots = []

        def snapshot(stage):
            snapshots.append(dict(stage=stage, **sample_memory()))

        snapshot("before-model")
        monitor.phase = "CPU-initialization"
        model = fixture.build().double()
        named = fixture.named(model)
        assert (
            set(named) == set(entries)
            and sum(p.numel() for p in named.values()) == PARAMETERS
        )
        assert (
            model.get_input_embeddings().weight is model.get_output_embeddings().weight
        )
        assert not model.is_gradient_checkpointing
        snapshot("after-model")
        directory = args.reference.parent

        def check_initial():
            hashes = {}
            for name, p in named.items():
                entry = entries[name]
                assert (
                    p.device.type == "cpu"
                    and p.dtype == torch.float64
                    and list(p.shape) == entry["shape"]
                )
                expected = torch.frombuffer(
                    read_blob(directory, entry["initial"]), dtype=torch.float32
                ).reshape(entry["shape"])
                assert torch.equal(p.detach(), expected.double())
                actual = p.detach().float().contiguous()
                assert torch.equal(actual, expected)
                hashes[name] = tensor_sha(actual)
                assert hashes[name] == entry["initial"]["sha256"]
                del expected, actual
            return hashes

        initial_hashes = check_initial()
        snapshot("after-initial-reference-check")
        completion = Completion(entries)
        hook_counts = {name: 0 for name in entries}
        gradient_hashes = {}
        maximum_visible_gradient_elements = 0
        maximum_live_scratch_bytes = 0
        in_hook = False
        scratch = args.output.parent / "gradient-scratch"
        if args.variant == "stream":
            scratch.mkdir()

        def check_gradient(name, p):
            completion.accept(name)
            entry = entries[name]
            assert (
                p.grad is not None
                and p.grad.device.type == "cpu"
                and p.grad.dtype == torch.float64
            )
            assert list(p.grad.shape) == entry["shape"] and torch.isfinite(p.grad).all()
            raw = p.grad.detach().contiguous()
            expected = torch.frombuffer(
                read_blob(directory, entry["gradient"]), dtype=torch.float64
            ).reshape(entry["shape"])
            assert torch.equal(raw, expected)
            pin = tensor_sha(raw)
            assert pin == entry["gradient"]["sha256"]
            gradient_hashes[name] = pin
            if args.variant == "stream":
                item = write_blob(scratch, "gradient.raw", memoryview(raw.numpy()))
                assert item["bytes"] <= bound and item["sha256"] == pin
                roundtrip = torch.frombuffer(
                    read_blob(scratch, item), dtype=torch.float64
                ).reshape(entry["shape"])
                assert torch.equal(roundtrip, raw)
                del roundtrip
                (scratch / item["file"]).unlink()
            del raw, expected

        if args.variant == "stream":

            def make_hook(name):
                def hook(p):
                    nonlocal maximum_visible_gradient_elements, maximum_live_scratch_bytes, in_hook
                    assert (
                        not in_hook
                    ), "Reentrant/concurrent hook would violate one-tensor scratch"
                    in_hook = True
                    try:
                        hook_counts[name] += 1
                        maximum_visible_gradient_elements = max(
                            maximum_visible_gradient_elements,
                            sum(
                                q.grad.numel()
                                for q in named.values()
                                if q.grad is not None
                            ),
                        )
                        maximum_live_scratch_bytes = max(
                            maximum_live_scratch_bytes, p.numel() * p.element_size()
                        )
                        check_gradient(name, p)
                        p.grad = None
                    finally:
                        in_hook = False

                return hook

            handles = [
                p.register_post_accumulate_grad_hook(make_hook(name))
                for name, p in named.items()
            ]
        monitor.phase = "CPU-forward"
        ids, labels = fixture.data(0)
        value = fixture.loss(
            model(input_ids=ids).logits,
            labels,
            CONFIG["global_batch"] * (CONFIG["tokens"] - 1),
        )
        loss = float(value.detach())
        assert math.isfinite(loss) and loss == manifest["CPU_FP64_loss"]
        snapshot("after-forward")
        monitor.phase = "CPU-backward-" + args.variant
        value.backward()
        del value
        retained = sum(p.grad.numel() for p in named.values() if p.grad is not None)
        snapshot("after-backward")
        if args.variant == "retained":
            assert retained == PARAMETERS
            monitor.phase = "CPU-retained-gradient-reference-check"
            for name, p in named.items():
                check_gradient(name, p)
        else:
            assert retained == 0 and all(count == 1 for count in hook_counts.values())
            assert maximum_visible_gradient_elements <= bound // 8
            assert maximum_live_scratch_bytes == bound and not list(scratch.iterdir())
            scratch.rmdir()
        completion.finish()
        for handle in handles:
            handle.remove()
        handles = []
        assert check_initial() == initial_hashes
        snapshot("after-reference-check")
        assert not torch.cuda.is_initialized()
        after = sample_memory()["events"]
        assert all(before[k] == after[k] == 0 for k in ["oom", "oom_kill"])
        result.update(
            status="passed",
            CPU_FP64_loss=loss,
            gradient_parameters=len(entries),
            gradient_elements=PARAMETERS,
            initial_sha256=initial_hashes,
            gradient_sha256=gradient_hashes,
            hook_counts=hook_counts,
            retained_gradient_elements_after_backward=retained,
            maximum_visible_gradient_elements_at_hook=maximum_visible_gradient_elements,
            maximum_live_scratch_bytes=maximum_live_scratch_bytes,
            scratch_removed=True,
            snapshots=snapshots,
            cgroup_events_after=after,
            worker_seconds=time.perf_counter() - start,
            checks=[
                "CPU-only-exact-loss-initial-parameters",
                "ALL-FP64-gradient-bits-match-D35",
                "complete-once-per-parameter-validation",
                "bounded-scratch-roundtrip-preservation",
                "retained-or-streamed-gradient-state-contract",
                "CPU-memory-swap-OOM-observations",
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
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
