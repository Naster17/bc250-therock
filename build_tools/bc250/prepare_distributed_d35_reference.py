#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only FP64 oracle, exact raw-blob roundtrip and separately admitted storage."""
import argparse
import math
import os
from pathlib import Path
import shutil
import time
import uuid
import torch
import fixture_d35 as fixture
from capacity_plan_d35 import CONFIG, PARAMETERS, GIB
from distributed_d32_capacity import meminfo
from distributed_d33_fresh import Monitor, sample_memory, package_identity
from distributed_d05_train import atomic_json
from artifact_d35 import (
    admit_storage,
    digest,
    source_hashes,
    write_blob,
    read_blob,
    load_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        status="running",
        phase="reference",
        config=CONFIG,
        sources=source_hashes(),
        checks=[],
    )
    monitor = None
    start = time.perf_counter()
    try:
        package_identity()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        assert not torch.cuda.is_initialized()
        assert meminfo()["MemAvailable"] >= 12 * GIB
        assert len(os.sched_getaffinity(0)) <= 4
        result["storage"] = admit_storage(
            shutil.disk_usage(args.output.parent).free, PARAMETERS, producing=True
        )
        directory = args.output.parent / "reference"
        directory.mkdir()
        result.update(
            torch=torch.__version__,
            physical_rank=int(os.environ["RANK"]),
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
        )
        monitor = Monitor()
        before = sample_memory()["events"]
        result["cgroup_events_before"] = before
        monitor.phase = "CPU-FP64-oracle"
        model = fixture.build().double()
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS
        assert (
            model.get_input_embeddings().weight is model.get_output_embeddings().weight
        )
        ids, labels = fixture.data(0)
        value = fixture.loss(
            model(input_ids=ids).logits,
            labels,
            CONFIG["global_batch"] * (CONFIG["tokens"] - 1),
        )
        value.backward()
        loss = float(value.detach())
        assert math.isfinite(loss)
        del value
        monitor.phase = "stream-reference-write-and-roundtrip"
        entries = []
        for index, (name, p) in enumerate(fixture.named(model).items()):
            assert p.device.type == "cpu" and p.dtype == p.grad.dtype == torch.float64
            assert torch.isfinite(p.grad).all()
            initial = p.detach().float().contiguous()
            gradient = p.grad.detach().contiguous()
            entry = dict(
                name=name,
                shape=list(p.shape),
                initial=write_blob(
                    directory, f"{index:03d}-initial.raw", memoryview(initial.numpy())
                ),
                gradient=write_blob(
                    directory, f"{index:03d}-gradient.raw", memoryview(gradient.numpy())
                ),
            )
            for key, expected, dtype in [
                ("initial", initial, torch.float32),
                ("gradient", gradient, torch.float64),
            ]:
                actual = torch.frombuffer(
                    read_blob(directory, entry[key]), dtype=dtype
                ).reshape(entry["shape"])
                assert torch.equal(actual, expected)
                del actual
            entries.append(entry)
            del initial, gradient
        manifest = dict(
            schema=1,
            status="passed",
            config=CONFIG,
            parameters=PARAMETERS,
            sources=result["sources"],
            torch=torch.__version__,
            cuda_initialized=False,
            CPU_FP64_loss=loss,
            tensors=entries,
        )
        manifest_path = directory / "manifest.json"
        atomic_json(manifest_path, manifest)
        pin = digest(manifest_path)
        assert load_manifest(manifest_path, pin, CONFIG, PARAMETERS) == manifest
        assert not torch.cuda.is_initialized()
        after = sample_memory()["events"]
        assert all(before[k] == after[k] == 0 for k in ["oom", "oom_kill"])
        result.update(
            status="passed",
            manifest_sha256=pin,
            CPU_FP64_loss=loss,
            artifact_bytes=sum(
                e[k]["bytes"] for e in entries for k in ["initial", "gradient"]
            ),
            gradient_parameters=len(entries),
            gradient_elements=PARAMETERS,
            cgroup_events_after=after,
            worker_seconds=time.perf_counter() - start,
            checks=[
                "CPU-only-FP64-loss-ALL-gradients",
                "ALL-artifact-bitexact-roundtrip",
                "bounded-file-hashes-source-shape-coverage",
                "storage-reserve-no-global-cache-flush",
            ],
        )
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if monitor is not None:
            result["memory_samples"] = monitor.finish()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
