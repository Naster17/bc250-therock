#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fresh D35 eight-update measurement bound to the D42 shape/rank/math proof."""
import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import time
import distributed_d35_growth as base
import artifact_d35 as artifact
from fixture_d41 import select
from capacity_plan_d41 import CONFIG, CONTROL, parameters
from capacity_plan_d42 import descriptor, admit
from qualification_d43 import proof, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["control", "large"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    rank = int(os.environ["RANK"])
    item = proof(a.shape, rank)
    reference = item["worker"]
    metadata = descriptor(a.shape)
    base.CONFIG = dict(CONTROL if a.shape == "control" else CONFIG)
    base.PARAMETERS = parameters(base.CONFIG)
    select(a.shape)
    assert not base.torch.cuda.is_initialized()
    cache_events, norms = [], []
    optimizer = None
    updates = 0
    native_backward, native_optimizer = (
        base.fixture.gpu_backward,
        base.fixture.optimizer,
    )
    native_clip, native_publish = base.clip, base.atomic_json

    def trim(stage):
        base.torch.cuda.synchronize()
        before = dict(
            allocated_bytes=base.torch.cuda.memory_allocated(),
            reserved_bytes=base.torch.cuda.memory_reserved(),
        )
        base.torch.cuda.empty_cache()
        after = dict(
            allocated_bytes=base.torch.cuda.memory_allocated(),
            reserved_bytes=base.torch.cuda.memory_reserved(),
        )
        assert before["allocated_bytes"] == after["allocated_bytes"]
        assert after["reserved_bytes"] <= before["reserved_bytes"]
        cache_events.append(dict(stage=stage, before=before, after=after))

    def backward(model, index, physical_rank, world, traffic):
        trim("before-backward-" + str(index))
        result = native_backward(model, index, physical_rank, world, traffic)
        trim("after-backward-" + str(index))
        return result

    def make_optimizer(model):
        nonlocal optimizer
        optimizer = native_optimizer(model)
        step = optimizer.step

        def update(*args, **kwargs):
            nonlocal updates
            value = step(*args, **kwargs)
            updates += 1
            print("D43 completed update", updates, flush=True)
            return value

        optimizer.step = update
        return optimizer

    def clip(model):
        norm = native_clip(model)
        value = float(norm)
        assert math.isfinite(value) and value >= 0
        norms.append(value)
        return norm

    def load_metadata(path, pin, config, count):
        assert pin == metadata["manifest_sha256"]
        assert config == base.CONFIG and count == base.PARAMETERS
        # Metadata only: all raw access below is forbidden, including inherited routes.
        return metadata["manifest"]

    def load_proof(path, pin, mode, physical_rank):
        assert pin == item["sha256"] and mode == "fsdp" and physical_rank == rank
        return reference

    def finite_state():
        start = time.perf_counter()
        counts = {k: 0 for k in ["parameters", "gradients", "exp_avg", "exp_avg_sq"]}
        for group in optimizer.param_groups:
            for p in group["params"]:
                if not p.numel():
                    continue
                assert (
                    p.grad is not None and p.is_contiguous() and p.grad.is_contiguous()
                )
                state = optimizer.state[p]
                assert int(state["step"]) == updates == 8
                for key, value in [
                    ("parameters", p),
                    ("gradients", p.grad),
                    ("exp_avg", state["exp_avg"]),
                    ("exp_avg_sq", state["exp_avg_sq"]),
                ]:
                    assert value.is_cuda and value.dtype == base.torch.float32
                    flat = value.detach().view(-1)
                    for offset in range(0, flat.numel(), 1024**2):
                        assert bool(
                            base.torch.isfinite(flat[offset : offset + 1024**2]).all()
                        )
                    counts[key] += value.numel()
        assert set(counts.values()) == {base.PARAMETERS // 2}
        return dict(
            elements=counts,
            counter=updates,
            max_chunk_elements=1024**2,
            seconds=time.perf_counter() - start,
            outside_timed_updates=True,
        )

    def publish(path, record):
        record.update(
            shape=a.shape,
            qualification_descriptor_sha256=artifact.digest(
                Path(__file__).with_name("qualification_d43.json")
            ),
            raw_reference_reads=0,
            cache_trim_events=cache_events,
            completed_updates=updates,
            finite_gradient_norms=norms,
            full_audits_forbidden=True,
        )
        if record["status"] == "passed":
            record["final_GPU_finite_state"] = finite_state()
            record["checks"].append("ALL-final-local-GPU-state-finite-eight-counters")
        native_publish(path, record)

    # Fail if a future inherited path accidentally enters qualification or raw I/O.
    for name in ["load_manifest", "read_blob", "write_blob", "verify_blob"]:
        setattr(artifact, name, base.forbidden_audit)
    base.qualify = base.forbidden_audit
    base.read_blob = base.write_blob = base.tensor_blob = base.forbidden_audit
    base.load_manifest, base.load_proof = load_metadata, load_proof
    base.sources = lambda: sources(a.shape, rank)
    base.admit = lambda mode, available, device: admit(
        a.shape, available, shutil.disk_usage(a.output.parent).free, device_free=device
    )
    base.fixture.gpu_backward, base.fixture.optimizer = backward, make_optimizer
    base.clip, base.atomic_json = clip, publish
    sys.argv = [
        sys.argv[0],
        "--mode",
        "fsdp",
        "--phase",
        "measure",
        "--reference",
        str(a.output.parent / "no-raw-reference"),
        "--reference-sha256",
        metadata["manifest_sha256"],
        "--qualification",
        str(Path(__file__).with_name("qualification_d43.json")),
        "--qualification-sha256",
        item["sha256"],
        "--output",
        str(a.output),
    ]
    base.main()


if __name__ == "__main__":
    main()
