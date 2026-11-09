#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D32 reference, memory, matched timing and actual state-sharding evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile
from export_distributed_d32_on_host import LABELS
from capacity_plan_d32 import CONFIG, PARAMETERS, GIB


# Training source identities are recorded by each worker, independent of current files.
def read(path, host_rank):
    prefix = "home/nik/bc250work/evidence/distributed-d32-r1/"
    source = "home/nik/bc250work/tools/distributed-d32/"
    with tarfile.open(path) as archive:

        def record(name):
            return json.load(archive.extractfile(prefix + name))

        h = record("handoff.json")
        assert h["status"] == "passed"
        rows = []
        for label in LABELS:
            w, f = record(label + "/rank.json"), record(label + "/final.json")
            assert h["records"][label] == dict(final=f, worker=w) and f["worker"] == w
            assert (
                w["status"] == f["status"] == "passed"
                and f["mode"] == w["mode"] == label.split("-")[0]
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
            assert len(w["checks"]) == 6 and w["config"] == CONFIG
            assert (
                w["reference"]["gradient_elements"] == PARAMETERS
                and w["reference"]["gradient_parameters"] == 76
            )
            assert w["reference"]["ALL_AdamW_parameters_moments_counters"]
            assert abs(
                w["reference"]["loss"] - w["reference"]["CPU_FP64_loss"]
            ) <= 3e-6 + 5e-5 * abs(w["reference"]["CPU_FP64_loss"])
            for filename, key in [
                ("distributed_d32_capacity.py", "source_sha256"),
                ("capacity_plan_d32.py", "plan_source_sha256"),
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
            assert w["plan"]["measured_larger_fit"] is False
            assert len(w["samples"]) == 5 and w[
                "median_update_seconds"
            ] == statistics.median(s["seconds_max_rank"] for s in w["samples"])
            assert all(
                w["cgroup_events_before"].get(k, 0)
                == w["cgroup_events_after"].get(k, 0)
                for k in ["oom", "oom_kill"]
            )
            for sample in w["samples"]:
                assert sample["seconds_max_rank"] >= sample["seconds_local"] > 0
                assert (
                    math.isfinite(sample["global_loss"])
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
                    assert sample["traffic_input_bytes"]["reductions"] == PARAMETERS * 4
                    assert sample["traffic_input_bytes"]["gathers"] > 0
                    assert sample["traffic_input_bytes"]["DDP"] == 0
            trace = w["memory_samples"]
            assert trace and set(s["phase"] for s in trace) >= {
                "CPU-reference-and-audits",
                "measured-training",
            }
            measured = [s for s in trace if s["phase"] == "measured-training"]
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
                    whole_validation_cgroup_peak_MiB=max(
                        s["memory.peak"] for s in w["samples"]
                    )
                    / 1024**2,
                    whole_host_min_available_GiB=min(
                        s["host_available_bytes"] for s in trace
                    )
                    / GIB,
                    measured_host_min_available_GiB=min(
                        s["host_available_bytes"] for s in measured
                    )
                    / GIB,
                    host_swap_max_bytes=max(s["host_swap_used_bytes"] for s in trace),
                    cgroup_swap_max_bytes=max(
                        s.get("memory.swap.current", 0) for s in trace
                    ),
                    guarded_phase_seconds=f["guarded_phase_seconds"],
                    worker_seconds=w["worker_seconds"],
                    training_segment_seconds=w["training_segment_seconds"],
                    master_elements=expected,
                    AdamW_state_bytes=w["state"]["AdamW_state_bytes"],
                    collective_input_bytes_per_update=w["samples"][0][
                        "traffic_input_bytes"
                    ],
                )
            )
        return h, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host104", type=Path, required=True)
    p.add_argument("--host105", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    left, lr = read(a.host104, 0)
    right, rr = read(a.host105, 1)
    for label in LABELS:
        lw, rw = left["records"][label]["worker"], right["records"][label]["worker"]
        for l, r in zip(lw["samples"], rw["samples"]):
            if not label.startswith("single"):
                assert (
                    l["seconds_max_rank"] == r["seconds_max_rank"]
                    and l["global_loss"] == r["global_loss"]
                )
        # Same deterministic fixture and update sequence, without requiring cross-reduction bitwise identity.
        baseline = left["records"]["single-a"]["worker"]["samples"]
        for sequence in [lw["samples"], rw["samples"]]:
            assert all(
                abs(x["global_loss"] - y["global_loss"]) <= 1e-5
                for x, y in zip(sequence, baseline)
            )
    result = dict(
        status="passed",
        groups=72,
        guards=12,
        config=CONFIG,
        parameters=PARAMETERS,
        precision="FP32 all-trainable masters/compute/reductions; native AdamW",
        global_batch=8,
        microbatch=1,
        reference_updates_per_worker=1,
        warmup_updates=2,
        timed_updates=5,
        rows=lr + rr,
        native_runtime_changed=False,
        larger_than_single_fit_qualified=False,
        PLAN05_complete=False,
        boot_ids=[left["boot_id"], right["boot_id"]],
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
