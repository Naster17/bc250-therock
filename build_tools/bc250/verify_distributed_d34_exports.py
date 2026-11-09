#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay D34 source/guard/proof/state checks and matched loss/memory measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
from capacity_plan_d34 import CONFIG, PARAMETERS, GIB
from export_distributed_d34_on_host import LABELS


# Immutable qualification identities; historical replay never follows mutable sources.
SOURCE_SHA256 = {
    "distributed_d34_growth.py": "6d72b9afff02fc307c150c62f47b736ec79e9c2b006987ac6e451134a1ee7dc9",
    "fixture_d34.py": "c1875a7c386fc8683e5e45f8fe92a49f2dbc419615a39a54d9e3eefcf16ef2f1",
    "capacity_plan_d34.py": "6ebdc2de8e799857b0af6f46b66cd19a7c6c8376d3753e6ec6c5197ba257d495",
    "distributed_d32_capacity.py": "f6e78720483d065c55a55e37df6cc604dd8bfc2df5e4f408ae6144b32af225ab",
    "distributed_d33_fresh.py": "72ca07b27d9b23d84865f05505c2d6212d401bdace9acdc9df574252ad64b1d4",
    "capacity_plan_d32.py": "8c7687e47917d79c6433322f296493a98ea7bcd555333a0a27bdbae4985f4bbc",
    "qualification_d33.py": "7b20334af874b9cbaa656a28118639719bc87c3a88c0521682a02f4b0f5f3a10",
    "distributed_d05_train.py": "a598b26b0c30ddd1a6efe205a4e28bce8a03f1f69467aa0bdf82fbe7c7e750bb",
}


def read(path, rank):
    prefix = "home/nik/bc250work/evidence/distributed-d34-r1/"
    source = "home/nik/bc250work/tools/distributed-d34/"
    with tarfile.open(path) as archive:

        def blob(name):
            return archive.extractfile(prefix + name).read()

        def record(name):
            return json.loads(blob(name))

        h = record("handoff.json")
        rows = []
        uuids, cids = set(), set()
        for label in LABELS:
            w, f = record(label + "/rank.json"), record(label + "/final.json")
            assert h["records"][label] == dict(worker=w, final=f) and f["worker"] == w
            assert w["status"] == f["status"] == "passed"
            assert w["config"] == CONFIG and w["physical_rank"] == rank
            assert (
                f["kernel_log_unchanged"]
                and f["registry_unchanged"]
                and f["agent_shutdown_clean"]
            )
            assert f["docker_containers"] == dict(before=[], after=[])
            assert blob(label + "/kernel-before.log") == blob(
                label + "/kernel-after.log"
            )
            state, ack = record(label + "/guard/state.json"), record(
                label + "/guard/ack.json"
            )
            assert state["status"] == "acknowledged"
            assert all(state[k] == ack[k] for k in ["nonce", "phase", "boot_id"])
            assert state["boot_id"] == h["boot_id"] == f["boot_id"]
            assert w["birth"]["uuid"] not in uuids and f["container_id"] not in cids
            uuids.add(w["birth"]["uuid"])
            cids.add(f["container_id"])
            for filename, wanted in w["sources"].items():
                data = archive.extractfile(source + filename).read()
                assert hashlib.sha256(data).hexdigest() == wanted
            assert w["sources"] == SOURCE_SHA256
            world = 1 if w["mode"] == "single" else 2
            assert w["world"] == world and w["rank"] == (0 if world == 1 else rank)
            assert w["state"]["full_parameter_elements"] == PARAMETERS
            assert (
                w["state"]["local_master_elements"]
                == w["state"]["AdamW_moment_elements_each"]
                == PARAMETERS // world
            )
            assert w["full_CPU_optimizer_or_export"] is False
            qualification = label.startswith("qualify-")
            assert w["CPU_reference_executed_in_this_worker"] == qualification
            assert len(w["checks"]) == (6 if qualification else 5)
            assert w["phase"] == ("qualify" if qualification else "measure")
            if qualification:
                ref = w["reference"]
                assert (
                    ref["gradient_elements"] == PARAMETERS
                    and ref["gradient_parameters"] == 148
                )
                assert (
                    ref["ALL_AdamW_parameters_moments_counters"]
                    and not ref["full_CPU_optimizer_or_export"]
                )
                assert ref["local_moment_elements_each"] == PARAMETERS // world
                assert (
                    ref["maximum_CPU_moment_elements"]
                    == 2 * CONFIG["vocab"] * CONFIG["width"]
                )
            else:
                qual = "qualify-" + w["mode"]
                proof = record(qual + "/rank.json")
                proof_sha = hashlib.sha256(blob(qual + "/rank.json")).hexdigest()
                assert (
                    w["qualification_sha256"] == f["qualification_sha256"] == proof_sha
                )
                assert (
                    w["birth"]["uuid"]
                    != proof["birth"]["uuid"]
                    == w["qualification_birth_uuid"]
                )
                assert w["sources"] == proof["sources"] and w["torch"] == proof["torch"]
                assert (
                    f["registry_sha256"]
                    == record(qual + "/final.json")["registry_sha256"]
                )
                assert (
                    abs(w["initial_global_loss"] - proof["reference"]["loss"]) <= 1e-5
                )
                assert len(w["samples"]) == 5
                assert w["median_update_seconds"] == statistics.median(
                    s["seconds_max_rank"] for s in w["samples"]
                )
                for sample in w["samples"]:
                    assert sample["seconds_max_rank"] >= sample["seconds_local"] > 0
                    assert (
                        all(
                            sample["traffic_input_bytes"][k] == 0
                            for k in ["gathers", "reductions"]
                        )
                        if world == 1
                        else (
                            sample["traffic_input_bytes"]["reductions"]
                            == PARAMETERS * 4
                            and sample["traffic_input_bytes"]["gathers"] > 0
                        )
                    )
            trace = w["memory_samples"]
            assert trace and min(s["host_available_bytes"] for s in trace) > 3 * GIB
            assert all(
                w["cgroup_events_before"][k] == w["cgroup_events_after"][k] == 0
                for k in ["oom", "oom_kill"]
            )
            assert max(s["host_swap_used_bytes"] for s in trace) == 0
            assert max(s["memory.swap.current"] for s in trace) == 0
            samples = (
                trace
                if qualification
                else [s for s in trace if s["phase"] == "measured-training"]
            )
            assert samples
            rows.append(
                dict(
                    host=rank,
                    label=label,
                    mode=w["mode"],
                    phase=w["phase"],
                    median_update_ms=(
                        None if qualification else 1000 * w["median_update_seconds"]
                    ),
                    peak_live_MiB=(
                        None
                        if qualification
                        else max(s["peak_allocated_bytes"] for s in w["samples"])
                        / 1024**2
                    ),
                    peak_reserved_MiB=(
                        None
                        if qualification
                        else max(s["peak_reserved_bytes"] for s in w["samples"])
                        / 1024**2
                    ),
                    sampled_cgroup_max_MiB=max(s["memory.current"] for s in samples)
                    / 1024**2,
                    whole_cgroup_peak_MiB=max(s["memory.peak"] for s in trace)
                    / 1024**2,
                    host_min_available_GiB=min(s["host_available_bytes"] for s in trace)
                    / GIB,
                    local_master_elements=PARAMETERS // world,
                    AdamW_state_bytes=w["state"]["AdamW_state_bytes"],
                    guarded_phase_seconds=f["guarded_phase_seconds"],
                )
            )
        return h, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host104", type=Path, required=True)
    parser.add_argument("--host105", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, lr = read(args.host104, 0)
    right, rr = read(args.host105, 1)
    all_workers = [
        h["records"][label]["worker"] for h in [left, right] for label in LABELS
    ]
    refs = [w["reference"]["loss"] for w in all_workers if w["phase"] == "qualify"]
    assert max(refs) - min(refs) <= 1e-5
    baseline = left["records"]["single-a"]["worker"]["samples"]
    for w in all_workers:
        if w["phase"] == "measure":
            assert all(
                abs(s["global_loss"] - b["global_loss"]) <= 1e-5
                for s, b in zip(w["samples"], baseline)
            )
    for label in ["fsdp-a", "fsdp-b"]:
        lw, rw = [h["records"][label]["worker"] for h in [left, right]]
        assert all(
            l["global_loss"] == r["global_loss"]
            and l["seconds_max_rank"] == r["seconds_max_rank"]
            for l, r in zip(lw["samples"], rw["samples"])
        )
    result = dict(
        status="passed",
        groups=sum(len(w["checks"]) for w in all_workers),
        guards=12,
        config=CONFIG,
        parameters=PARAMETERS,
        fresh_shape_specific_CPU_qualification=True,
        full_CPU_optimizer_or_export=False,
        qualified_trajectory_parity=True,
        larger_than_single_fit_qualified=False,
        native_runtime_changed=False,
        PLAN05_complete=False,
        rows=lr + rr,
        boot_ids=[left["boot_id"], right["boot_id"]],
        archives=[
            dict(
                path=str(path),
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in [args.host104, args.host105]
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
