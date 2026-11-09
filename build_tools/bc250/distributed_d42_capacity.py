#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D39 per-unit GPU math on source-bound shapes with CPU pre-PG readiness."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import time
import artifact_d35 as artifact
from artifact_d41 import sources as oracle_sources
import distributed_d35_growth as base
import distributed_d39_units as previous
from audit_chunks_d42 import Auditor
from cpu_adamw_chunks_d42 import equivalence_check
from capacity_plan_d42 import descriptor, admit, READINESS_SECONDS
from capacity_plan_d41 import CONFIG, CONTROL, parameters
from fixture_d41 import select
from reference_transport_d40 import RemoteStore

previous_sources = previous.sources


def sources():
    result = previous_sources()
    result.update(oracle_sources())
    result.update(
        {
            n: artifact.digest(Path(__file__).with_name(n))
            for n in [
                "reference_transport_d40.py",
                "capacity_plan_d42.py",
                "distributed_d42_capacity.py",
                "serve_reference_d42.py",
                "oracle_descriptor_d42.json",
                "audit_chunks_d42.py",
                "cpu_adamw_chunks_d42.py",
            ]
        }
    )
    return result


class StageAuditor(Auditor):
    def trim(self, stage):
        base.torch.cuda.synchronize()
        before = dict(
            allocated_bytes=base.torch.cuda.memory_allocated(),
            reserved_bytes=base.torch.cuda.memory_reserved(),
            host_available_bytes=base.meminfo()["MemAvailable"],
        )
        base.torch.cuda.empty_cache()
        after = dict(
            allocated_bytes=base.torch.cuda.memory_allocated(),
            reserved_bytes=base.torch.cuda.memory_reserved(),
            host_available_bytes=base.meminfo()["MemAvailable"],
        )
        self.cache_events.append(dict(stage=stage, before=before, after=after))
        print("D42 cache trim", stage, json.dumps(self.cache_events[-1]), flush=True)

    def qualify(self, *args, **kwargs):
        self.cache_events = []
        self.equivalence = equivalence_check()
        self.trim("after-wrap-before-backward")
        result = super().qualify(*args, **kwargs)
        result["cache_trim_events"] = self.cache_events
        result["CPU_AdamW_chunk_equivalence"] = self.equivalence
        return result

    def after_updated_audit(self):
        self.trim("before-local-moments")

    def parameters(self, model, entries, stage):
        self.trim(stage)
        print("D42 GPU audit stage:", stage, flush=True)
        yield from super().parameters(model, entries, stage)


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--shape", choices=["control", "large"], required=True)
    p.add_argument("--transport", choices=["local", "remote"], required=True)
    p.add_argument("--endpoint")
    p.add_argument("--ready-port", type=int, required=True)
    a, remaining = p.parse_known_args()
    assert 1024 <= a.ready_port <= 65535
    config = CONTROL if a.shape == "control" else CONFIG
    proof = descriptor(a.shape)
    base.CONFIG = dict(config)
    base.PARAMETERS = parameters(config)
    select(a.shape)
    path = Path(remaining[remaining.index("--reference") + 1])
    pin = remaining[remaining.index("--reference-sha256") + 1]
    assert pin == proof["manifest_sha256"]
    rank = int(os.environ["RANK"])
    assert a.transport == ("local" if rank == 0 else "remote")
    remote = None
    if rank == 1:
        assert a.endpoint == "http://192.168.1.104:29842"
        assert sorted(x.name for x in path.parent.iterdir()) == ["manifest.json"]
        remote = RemoteStore(a.endpoint, path, pin)
        artifact.verify_blob = remote.verify
        base.read_blob = remote.read
    else:
        assert a.endpoint is None
    store = None
    readiness = None

    def load(path, pin, config, count):
        nonlocal store, readiness
        start = time.monotonic()
        manifest = artifact.load_manifest(path, pin, config, count, check_sources=False)
        assert (
            manifest == proof["manifest"]
            and manifest["sources"] == proof["worker"]["sources"]
        )
        assert not base.torch.cuda.is_initialized()
        print("D42 CPU references verified; entering readiness barrier", flush=True)
        store = base.dist.TCPStore(
            "192.168.1.104",
            a.ready_port,
            2,
            rank == 0,
            timeout=timedelta(seconds=READINESS_SECONDS),
        )
        identity = dict(
            manifest_sha256=pin,
            shape=a.shape,
            sources=sources(),
            CUDA_initialized=False,
        )
        store.set(f"ready{rank}", json.dumps(identity, sort_keys=True))
        store.wait(["ready0", "ready1"], timedelta(seconds=READINESS_SECONDS))
        peers = [json.loads(store.get(f"ready{i}")) for i in range(2)]
        assert peers == [identity, identity] and not base.torch.cuda.is_initialized()
        readiness = dict(
            peers=peers,
            port=a.ready_port,
            CPU_only=True,
            timeout_seconds=READINESS_SECONDS,
            verification_and_wait_seconds=time.monotonic() - start,
            native_GPU_PG_timeout_seconds=120,
        )
        print(
            "D42 CPU readiness complete; entering native GPU initialization", flush=True
        )
        return manifest

    def admit_worker(mode, available, device):
        assert mode == "fsdp"
        return admit(
            a.shape, available, shutil.disk_usage(path.parent).free, device_free=device
        )

    original = base.atomic_json

    def publish(path, record):
        record["reference_transport"] = dict(
            kind=a.transport, stats=remote.stats if remote else None
        )
        record["readiness"] = readiness
        record["shape"] = a.shape
        original(path, record)

    base.load_manifest = load
    base.admit = admit_worker
    base.atomic_json = publish
    previous.sources = sources
    previous.Auditor = StageAuditor
    sys.argv = [sys.argv[0], "--audit", "reference", *remaining]
    previous.main()
    # Keep the standalone CPU store alive until both complete native PG teardown.
    store.set(f"done{rank}", "done")
    store.wait(["done0", "done1"], timedelta(seconds=120))
    if rank == 1:
        store.set("client-finished", "done")
    else:
        store.wait(["client-finished"], timedelta(seconds=120))
    del store


if __name__ == "__main__":
    main()
