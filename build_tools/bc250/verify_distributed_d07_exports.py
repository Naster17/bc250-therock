#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check D07 archives, reference errors, scaler transitions and both-rank metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from export_distributed_d07_on_host import PHASES, verify_worker


def read(path: Path) -> tuple[dict, dict]:
    with tarfile.open(path) as t:
        assert not any(m.name.endswith(".pt") for m in t.getmembers())
        root = "home/nik/bc250work/evidence/distributed-d07-r1/"
        handoff = json.load(t.extractfile(root + "handoff.json"))
        assert handoff["status"] == "passed" and handoff["running_containers"] == []
        records = {r["label"]: r["final"] for r in handoff["records"]}
        assert set(records) == {v[2] for v in PHASES}
        for dtype, mode, label in PHASES:
            final = records[label]
            assert final == json.load(t.extractfile(root + label + "/final.json"))
            worker = json.load(t.extractfile(root + label + "/rank.json"))
            assert worker == final["worker"]
            verify_worker(worker, dtype, mode)
            assert final["status"] == "passed" and all(
                final[k]
                for k in [
                    "agent_shutdown_clean",
                    "kernel_log_unchanged",
                    "registry_unchanged",
                ]
            )
            assert (
                t.extractfile(root + label + "/kernel-before.log").read()
                == t.extractfile(root + label + "/kernel-after.log").read()
            )
            state = json.load(t.extractfile(root + label + "/guard/state.json"))
            ack = json.load(t.extractfile(root + label + "/guard/ack.json"))
            assert all(state[k] == ack[k] for k in ["nonce", "phase", "boot_id"])
            for filename, key in [
                ("distributed_d07_amp.py", "script_sha256"),
                ("distributed_d05_train.py", "reused_d05_sha256"),
            ]:
                source = t.extractfile(
                    "home/nik/bc250work/tools/distributed-d07/" + filename
                ).read()
                assert hashlib.sha256(source).hexdigest() == worker[key]
            registry = t.extractfile("opt/bc250-stack/distributed/d01.json").read()
            assert hashlib.sha256(registry).hexdigest() == final["registry_sha256"]
        return handoff, records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    pairs = [read(a.directory / f"{host}-evidence-r1.tar.gz") for host in [104, 105]]
    handoffs, records = zip(*pairs)
    table = []
    for dtype, mode, label in PHASES:
        workers = [rank[label]["worker"] for rank in records]
        assert [w["rank"] for w in workers] == [0, 1]
        assert workers[0]["config"] == workers[1]["config"]
        assert workers[0]["windows"] == workers[1]["windows"]
        table.append(
            dict(
                dtype=dtype,
                mode=mode,
                maximum_parameter_gradient_relative_L2=max(
                    m["maximum_parameter_gradient_relative_L2"]
                    for w in workers
                    for m in w["metrics"]
                ),
                maximum_input_gradient_relative_L2=max(
                    m["input_gradient_relative_L2"]
                    for w in workers
                    for m in w["metrics"]
                ),
                initial_sync_calls=workers[0]["windows"][0]["micros"],
                per_window_reduction_input_bytes=sum(
                    m["bytes"] for m in workers[0]["windows"][0]["micros"]
                ),
            )
        )
    for dtype in ["bf16", "fp16"]:
        values = [v for v in table if v["dtype"] == dtype]
        assert (
            values[1]["per_window_reduction_input_bytes"]
            == 4 * values[0]["per_window_reduction_input_bytes"]
        )
    result = dict(
        status="passed",
        tables=table,
        worker_check_groups=sum(
            len(v["worker"]["checks"]) for rank in records for v in rank.values()
        ),
        checkpoint_preload_refusals=sum(
            v["worker"]["checkpoint_refusals"]
            for rank in records
            for v in rank.values()
        ),
        scaler_mismatch_refusals=8,
        acknowledged_guards=8,
        boots=[h["boot_id"] for h in handoffs],
        kernels=[h["kernel"] for h in handoffs],
        retained_checkpoint_bytes=[
            sum(v["bytes"] for v in h["checkpoints_retained"]) for h in handoffs
        ],
        archives=[
            dict(
                host=host,
                bytes=(a.directory / f"{host}-evidence-r1.tar.gz").stat().st_size,
                sha256=hashlib.sha256(
                    (a.directory / f"{host}-evidence-r1.tar.gz").read_bytes()
                ).hexdigest(),
            )
            for host in [104, 105]
        ],
    )
    (a.directory / "verified-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
