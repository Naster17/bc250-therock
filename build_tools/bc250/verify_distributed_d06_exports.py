#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independently verify both D06 metadata exports and summarize matched timings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

from run_distributed_d06_on_host import PHASES


def read_archive(path):
    with tarfile.open(path) as archive:
        assert not any(p.name.endswith(".pt") for p in archive.getmembers())
        members = [
            p
            for p in archive.getmembers()
            if p.name.endswith("distributed-d06-r1/handoff.json")
        ]
        assert len(members) == 1
        summary = json.load(archive.extractfile(members[0]))
        records = {r["label"]: r["final"] for r in summary["records"]}
        assert set(records) == {p[2] for p in PHASES}
        for label, final in records.items():
            assert final["status"] == "passed"
            assert all(
                final[k]
                for k in [
                    "agent_shutdown_clean",
                    "kernel_log_unchanged",
                    "registry_unchanged",
                ]
            )
            prefix = f"home/nik/bc250work/evidence/distributed-d06-r1/{label}/"
            assert json.load(archive.extractfile(prefix + "final.json")) == final
            assert (
                json.load(archive.extractfile(prefix + "rank.json")) == final["worker"]
            )
            before = archive.extractfile(prefix + "kernel-before.log").read()
            after = archive.extractfile(prefix + "kernel-after.log").read()
            assert before == after
            state = json.load(archive.extractfile(prefix + "guard/state.json"))
            ack = json.load(archive.extractfile(prefix + "guard/ack.json"))
            assert all(state[k] == ack[k] for k in ["nonce", "phase", "boot_id"])
            worker = final["worker"]
            source = archive.extractfile(
                "home/nik/bc250work/tools/distributed-d06/distributed_d06_accumulate.py"
            ).read()
            assert hashlib.sha256(source).hexdigest() == worker["script_sha256"]
            reused = archive.extractfile(
                "home/nik/bc250work/tools/distributed-d06/distributed_d05_train.py"
            ).read()
            assert hashlib.sha256(reused).hexdigest() == worker["reused_d05_sha256"]
        return summary, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    pairs = [
        read_archive(a.directory / f"{host}-evidence-r1.tar.gz") for host in [104, 105]
    ]
    summaries, records = zip(*pairs)
    tables = []
    for case in ["small", "larger"]:
        for order in ["a", "b"]:
            labels = {
                mode: f"{case}-{mode}-{order}" for mode in ["single", "sync", "no-sync"]
            }
            workers = {
                mode: [rank[label]["worker"] for rank in records]
                for mode, label in labels.items()
            }
            for mode in ["sync", "no-sync"]:
                left, right = workers[mode]
                assert left["rank"] == 0 and right["rank"] == 1
                assert (
                    left["config"] == right["config"] == workers["single"][0]["config"]
                )
                assert left["step_seconds"] == right["step_seconds"]
                for lstep, rstep in zip(left["traffic_steps"], right["traffic_steps"]):
                    assert lstep["bytes"] == rstep["bytes"]
            sync, nosync = workers["sync"][0], workers["no-sync"][0]
            traffic = {
                mode: worker[0]["traffic_steps"][4] for mode, worker in workers.items()
            }
            assert (
                traffic["sync"]["bytes"]
                == traffic["no-sync"]["bytes"] * sync["accumulation"]
            )
            tables.append(
                dict(
                    case=case,
                    order=order,
                    single_ms=[
                        v["median_step_seconds"] * 1000 for v in workers["single"]
                    ],
                    sync_ms=sync["median_step_seconds"] * 1000,
                    no_sync_ms=nosync["median_step_seconds"] * 1000,
                    sync_to_no_sync_speed_ratio=sync["median_step_seconds"]
                    / nosync["median_step_seconds"],
                    steady_bucket_traffic=traffic,
                )
            )
    result = dict(
        status="passed",
        tables=tables,
        worker_check_groups=sum(
            len(r["worker"]["checks"]) for rank in records for r in rank.values()
        ),
        preload_refusals=sum(
            r["worker"]["checkpoint_refusals"]
            for rank in records
            for r in rank.values()
        ),
        acknowledged_guards=sum(len(rank) for rank in records),
        boots=[s["boot_id"] for s in summaries],
        memory={
            label: [r[label]["worker"]["memory"] for r in records]
            for _, _, label in PHASES
        },
        retained_checkpoint_bytes=[
            sum(c["bytes"] for c in s["checkpoints_retained"]) for s in summaries
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
    print(json.dumps({k: v for k, v in result.items() if k != "memory"}, indent=2))


if __name__ == "__main__":
    main()
