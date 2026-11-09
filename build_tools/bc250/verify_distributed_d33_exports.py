#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify source-bound fresh memory/timing controls against published D32 evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
from capacity_plan_d32 import CONFIG, PARAMETERS, GIB
from qualification_d33 import PROOF_SHA256
from export_distributed_d33_on_host import LABELS


def read(path, host_rank):
    prefix = "home/nik/bc250work/evidence/distributed-d33-r1/"
    source = "home/nik/bc250work/tools/distributed-d33/"
    with tarfile.open(path) as archive:

        def record(name):
            return json.load(archive.extractfile(prefix + name))

        h = record("handoff.json")
        blob = archive.extractfile(
            source + f"distributed_d33_qualification_{host_rank}.json"
        ).read()
        assert hashlib.sha256(blob).hexdigest() == PROOF_SHA256[host_rank]
        proof = json.loads(blob)
        for name, wanted in proof["source_sha256"].items():
            assert (
                hashlib.sha256(archive.extractfile(source + name).read()).hexdigest()
                == wanted
            )
        rows, uuids = [], set()
        for label in LABELS:
            w, f = record(label + "/rank.json"), record(label + "/final.json")
            assert h["records"][label] == dict(final=f, worker=w) and f["worker"] == w
            assert (
                w["status"] == f["status"] == "passed"
                and w["mode"] == f["mode"] == label.split("-")[0]
            )
            assert (
                f["registry_unchanged"]
                and f["kernel_log_unchanged"]
                and f["agent_shutdown_clean"]
            )
            assert f["docker_containers"] == dict(before=[], after=[])
            assert (
                archive.extractfile(prefix + label + "/kernel-before.log").read()
                == archive.extractfile(prefix + label + "/kernel-after.log").read()
            )
            state, ack = record(label + "/guard/state.json"), record(
                label + "/guard/ack.json"
            )
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            assert state["boot_id"] == h["boot_id"]
            assert len(w["checks"]) == 7 and w["config"] == CONFIG
            assert (
                w["CPU_reference_executed_in_this_worker"] is False
                and w["full_state_audits_executed_in_this_worker"] is False
            )
            assert w["qualification"]["proof_sha256"] == PROOF_SHA256[host_rank]
            assert (
                w["qualification"]["archive_sha256"]
                == f["qualification_archive_sha256"]
                == proof["archive_sha256"]
            )
            qualified = proof["reference"][w["mode"]]
            assert f["registry_sha256"] == qualified["registry_sha256"]
            assert f["container_id"] not in proof["original_container_ids"]
            assert w["birth"]["uuid"] not in uuids
            uuids.add(w["birth"]["uuid"])
            for filename, key in [
                ("distributed_d33_fresh.py", "source_sha256"),
                ("qualification_d33.py", "qualification_helper_sha256"),
            ]:
                assert (
                    hashlib.sha256(
                        archive.extractfile(source + filename).read()
                    ).hexdigest()
                    == w[key]
                )
            world = 1 if w["mode"] == "single" else 2
            assert w["world"] == world and w["rank"] == (0 if world == 1 else host_rank)
            expected = PARAMETERS // world if w["mode"] == "fsdp" else PARAMETERS
            assert w["state"]["full_parameter_elements"] == PARAMETERS
            assert (
                w["state"]["local_master_elements"]
                == w["state"]["AdamW_moment_elements_each"]
                == expected
            )
            assert w["state"] == qualified["state"]
            assert (
                abs(w["initial_global_loss"] - qualified["reference"]["loss"]) <= 1e-5
            )
            assert len(w["samples"]) == 5 and w[
                "median_update_seconds"
            ] == statistics.median(s["seconds_max_rank"] for s in w["samples"])
            for sample, wanted_loss in zip(w["samples"], qualified["global_losses"]):
                assert sample["seconds_max_rank"] >= sample["seconds_local"] > 0
                assert (
                    abs(sample["global_loss"] - wanted_loss) <= 1e-5
                    and sample["host_available_bytes"] > 3 * GIB
                )
                assert sample["events"]["oom"] == sample["events"]["oom_kill"] == 0
                if w["mode"] == "single":
                    assert all(v == 0 for v in sample["traffic_input_bytes"].values())
                elif w["mode"] == "ddp":
                    assert sample["traffic_input_bytes"]["DDP"] == PARAMETERS * 4
                    assert (
                        sample["traffic_input_bytes"]["gathers"]
                        == sample["traffic_input_bytes"]["reductions"]
                        == 0
                    )
                else:
                    assert (
                        sample["traffic_input_bytes"]["reductions"] == PARAMETERS * 4
                        and sample["traffic_input_bytes"]["gathers"] > 0
                    )
                    assert sample["traffic_input_bytes"]["DDP"] == 0
            assert all(
                w["cgroup_events_before"][k] == w["cgroup_events_after"][k] == 0
                for k in ["oom", "oom_kill"]
            )
            snapshots = {s["stage"]: s for s in w["snapshots"]}
            assert set(snapshots) == {
                "before-model",
                "after-model-wrap",
                "after-initial-update",
                "after-owned-cache-release",
                "after-warmup",
                "after-measured-training",
            }
            trace = w["memory_samples"]
            assert "CPU-reference-and-audits" not in {s["phase"] for s in trace}
            measured = [s for s in trace if s["phase"] == "measured-training"]
            assert measured and min(s["host_available_bytes"] for s in trace) > 3 * GIB
            rows.append(
                dict(
                    host=host_rank,
                    label=label,
                    mode=w["mode"],
                    order=label[-1],
                    median_update_ms=1000 * w["median_update_seconds"],
                    peak_live_MiB=max(s["peak_allocated_bytes"] for s in w["samples"])
                    / 1024**2,
                    peak_reserved_MiB=max(
                        s["peak_reserved_bytes"] for s in w["samples"]
                    )
                    / 1024**2,
                    measured_cgroup_max_MiB=max(s["memory.current"] for s in measured)
                    / 1024**2,
                    whole_worker_cgroup_peak_MiB=max(
                        s["memory.peak"] for s in w["samples"]
                    )
                    / 1024**2,
                    measured_host_min_available_GiB=min(
                        s["host_available_bytes"] for s in measured
                    )
                    / GIB,
                    whole_host_min_available_GiB=min(
                        s["host_available_bytes"] for s in trace
                    )
                    / GIB,
                    host_swap_max_bytes=max(s["host_swap_used_bytes"] for s in trace),
                    cgroup_swap_max_bytes=max(s["memory.swap.current"] for s in trace),
                    measured_stat_max_MiB={
                        name: max(s["memory.stat"].get(name, 0) for s in measured)
                        / 1024**2
                        for name in [
                            "anon",
                            "file",
                            "kernel",
                            "sock",
                            "shmem",
                            "pagetables",
                        ]
                    },
                    RSS_after_measured_MiB=snapshots["after-measured-training"]["VmRSS"]
                    / 1024**2,
                    cgroup_initial_MiB=snapshots["before-model"]["memory.current"]
                    / 1024**2,
                    guarded_phase_seconds=f["guarded_phase_seconds"],
                    worker_seconds=w["worker_seconds"],
                    master_elements=expected,
                    AdamW_state_bytes=w["state"]["AdamW_state_bytes"],
                )
            )
        return h, proof, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host104", type=Path, required=True)
    p.add_argument("--host105", type=Path, required=True)
    p.add_argument("--baseline-results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    baseline = json.loads(a.baseline_results.read_text())
    assert baseline["status"] == "passed" and baseline["groups"] == 72
    left, lp, lr = read(a.host104, 0)
    right, rp, rr = read(a.host105, 1)
    for proof, published in zip([lp, rp], baseline["archives"]):
        assert proof["archive_sha256"] == published["sha256"]
    for label in LABELS:
        lw, rw = left["records"][label]["worker"], right["records"][label]["worker"]
        if not label.startswith("single"):
            assert all(
                l["seconds_max_rank"] == r["seconds_max_rank"]
                and l["global_loss"] == r["global_loss"]
                for l, r in zip(lw["samples"], rw["samples"])
            )
    rows = lr + rr
    for row, old in zip(rows, baseline["rows"]):
        assert (row["host"], row["label"]) == (old["host"], old["label"])
        row["D32_prior_cgroup_max_MiB"] = old["measured_cgroup_max_MiB"]
        row["cgroup_difference_from_D32_MiB"] = (
            row["measured_cgroup_max_MiB"] - old["measured_cgroup_max_MiB"]
        )
    result = dict(
        status="passed",
        groups=84,
        guards=12,
        config=CONFIG,
        parameters=PARAMETERS,
        CPU_reference_reused_from_D32=True,
        fresh_CPU_reference_executed=False,
        fresh_full_state_audits_executed=False,
        qualified_trajectory_global_loss_parity=True,
        rows=rows,
        native_runtime_changed=False,
        larger_than_single_fit_qualified=False,
        PLAN05_complete=False,
        boot_ids=[left["boot_id"], right["boot_id"]],
        baseline_archives=baseline["archives"],
        archives=[
            dict(
                path=str(path),
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in [a.host104, a.host105]
        ],
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    for row in rows:
        print(
            row["host"],
            row["label"],
            round(row["median_update_ms"], 2),
            round(row["peak_live_MiB"], 2),
            round(row["measured_cgroup_max_MiB"], 2),
            row["measured_stat_max_MiB"],
        )


if __name__ == "__main__":
    main()
