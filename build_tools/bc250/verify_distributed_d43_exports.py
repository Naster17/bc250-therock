#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay D42 prerequisite and independently check D43 training/resource evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile
from uuid import UUID
from verify_distributed_d42_exports import predecessors, read as read42
from capacity_plan_d41 import CONFIG, CONTROL, parameters, GIB
from capacity_plan_d42 import admit, headroom, HOST_REGISTRY_SHA256, GUARD_SECONDS
from qualification_d43 import DESCRIPTOR_SHA256

PINS = {
    "distributed_d43_train.py": "ed33832f4884ddc816135aa82ef7b8b4cd10dd7372907e556a21762831675488",
    "qualification_d43.py": "ac59bfbeb085199ef29db1156418f6e921451327ffaac1d57d565815795bd9cf",
    "qualification_d43.json": "1d8fef22942169acbd759d3639a9730d4090dd68ebf211debb660aea3121e7e1",
    "guard_distributed_d43_on_host.py": "71a00d80d3bcc108ac416f5a91699b672ba3409999931b38226fcdf76a2bc348",
    "distributed_d43_bounded_launch.py": "67f097ce3ce7bc2cf7ad6cf11a98d508f09111b65f4790b77ba14b6b76425cbb",
    "export_distributed_d43_on_host.py": "04a7ac9d9350839961f4fea9392bb265deeded0769498776ea7512421d4ee9c9",
}
LABELS = ["control", "large-training"]


def prerequisite(workspace):
    report = json.loads(
        (
            workspace
            / "therock/docs/development/bc250_distributed_d42_results/verified-results.json"
        ).read_text()
    )
    prior, oracle = predecessors(workspace)
    workers = {}
    for rank, entry in enumerate(report["archives"]):
        path = workspace / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        _, w, _ = read42(path, rank, prior[rank], oracle)
        workers[rank] = w
    return report, workers


def read(path, rank, prior):
    report, old = prior
    root = "home/nik/bc250work/evidence/distributed-d43-r1/"
    source = "home/nik/bc250work/tools/distributed-d43/"
    with tarfile.open(path) as archive:

        def blob(n):
            return archive.extractfile(root + n).read()

        def record(n):
            return json.loads(blob(n))

        assert not any(m.name.endswith(".raw") for m in archive.getmembers())
        for name, pin in PINS.items():
            assert (
                hashlib.sha256(archive.extractfile(source + name).read()).hexdigest()
                == pin
            )
        proof_bytes = archive.extractfile(source + "qualification_d43.json").read()
        assert hashlib.sha256(proof_bytes).hexdigest() == DESCRIPTOR_SHA256
        descriptor = json.loads(proof_bytes)
        assert descriptor["report"] == report
        h = record("handoff.json")
        assert h["status"] == "passed" and h["rank"] == rank
        assert str(UUID(h["boot_id"])) == h["boot_id"]
        assert not h["idle"]["docker_containers"] and not h["idle"]["podman_containers"]
        assert h["idle"]["watchdog"] == "active"
        registry = archive.extractfile("opt/bc250-stack/distributed/d01.json").read()
        assert hashlib.sha256(registry).hexdigest() == HOST_REGISTRY_SHA256[rank]
        rows = []
        births = set()
        containers = set()
        workers = {}
        for label, shape, config in zip(
            LABELS, ["control", "large"], [CONTROL, CONFIG]
        ):
            w, f = record(label + "/rank.json"), record(label + "/final.json")
            item = descriptor["proofs"][f"{rank}-{shape}"]
            old_label = "control" if shape == "control" else "large-audit"
            assert item["worker"] == old[rank][old_label]
            assert w["status"] == f["status"] == "passed" and f["worker"] == w
            assert (
                w["phase"] == f["phase"] == "measure"
                and w["mode"] == f["mode"] == "fsdp"
            )
            assert w["config"] == config and w["shape"] == f["shape"] == shape
            assert (
                w["rank"] == w["physical_rank"] == f["rank"] == rank and w["world"] == 2
            )
            assert w["torch"] == item["worker"]["torch"]
            expected = {
                **item["worker"]["sources"],
                **{
                    n: PINS[n]
                    for n in [
                        "qualification_d43.json",
                        "qualification_d43.py",
                        "distributed_d43_train.py",
                    ]
                },
            }
            assert w["sources"] == expected
            for name, pin in expected.items():
                assert (
                    hashlib.sha256(
                        archive.extractfile(source + name).read()
                    ).hexdigest()
                    == pin
                )
            assert (
                w["qualification_sha256"]
                == f["D42_qualification_sha256"]
                == item["sha256"]
            )
            assert w["qualification_birth_uuid"] == item["worker"]["birth"]["uuid"]
            assert (
                w["birth"]["uuid"] not in births
                and w["birth"]["uuid"] != w["qualification_birth_uuid"]
            )
            births.add(w["birth"]["uuid"])
            assert f["container_id"] not in containers
            containers.add(f["container_id"])
            assert f["kernel"] == h["kernel"] and f["boot_id"] == h["boot_id"]
            assert (
                f["agent_shutdown_clean"]
                and f["kernel_log_unchanged"]
                and f["registry_unchanged"]
            )
            assert f["registry_sha256"] == HOST_REGISTRY_SHA256[rank]
            assert f["docker_containers"] == dict(before=[], after=[])
            assert blob(label + "/kernel-before.log") == blob(
                label + "/kernel-after.log"
            )
            state, ack = record(label + "/guard/state.json"), record(
                label + "/guard/ack.json"
            )
            assert (
                state["status"] == "acknowledged"
                and state["timeout_seconds"] == GUARD_SECONDS
            )
            assert all(state[k] == ack[k] for k in ["nonce", "phase", "boot_id"])
            assert state["boot_id"] == f["boot_id"]
            assert w["raw_reference_reads"] == 0 and w["full_audits_forbidden"]
            assert (
                w["CPU_reference_executed_in_this_worker"] is False
                and w["full_CPU_optimizer_or_export"] is False
            )
            assert w["completed_updates"] == 8 and len(w["finite_gradient_norms"]) == 8
            assert all(math.isfinite(v) and v >= 0 for v in w["finite_gradient_norms"])
            finite = w["final_GPU_finite_state"]
            count = parameters(config)
            assert finite["elements"] == dict.fromkeys(
                ["parameters", "gradients", "exp_avg", "exp_avg_sq"], count // 2
            )
            assert (
                finite["counter"] == 8
                and finite["outside_timed_updates"]
                and finite["max_chunk_elements"] == 1024**2
            )
            assert (
                w["state"]["full_parameter_elements"] == count
                and w["state"]["local_master_elements"]
                == w["state"]["AdamW_moment_elements_each"]
                == count // 2
            )
            assert len(w["cache_trim_events"]) == 16
            for index, event in enumerate(w["cache_trim_events"]):
                assert event["stage"] == (
                    "before" if index % 2 == 0 else "after"
                ) + "-backward-" + str(index // 2)
                assert (
                    event["before"]["allocated_bytes"]
                    == event["after"]["allocated_bytes"]
                )
                assert (
                    event["after"]["reserved_bytes"]
                    <= event["before"]["reserved_bytes"]
                )
            assert (
                abs(w["initial_global_loss"] - item["worker"]["reference"]["loss"])
                <= 1e-5
            )
            assert len(w["samples"]) == 5 and len(w["checks"]) == 6
            for s in w["samples"]:
                assert (
                    math.isfinite(s["global_loss"])
                    and s["seconds_max_rank"] >= s["seconds_local"] > 0
                )
                assert s["host_available_bytes"] >= 3 * GIB
                assert set(s["traffic_input_bytes"]) == {"gathers", "reductions"}
                assert all(v > 0 for v in s["traffic_input_bytes"].values())
            assert w["median_update_seconds"] == statistics.median(
                s["seconds_max_rank"] for s in w["samples"]
            )
            admission = f["admission"]
            assert (
                admit(
                    shape,
                    admission["available_before_bytes"],
                    admission["free_disk_before_bytes"],
                )
                == admission
            )
            plan = w["plan"]
            assert (
                admit(
                    shape,
                    plan["available_before_bytes"],
                    plan["free_disk_before_bytes"],
                    device_free=plan["device_free_before_bytes"],
                )
                == plan
            )
            for s in f["headroom_samples"]:
                headroom(s["memory_bytes"], s["available_bytes"], s["free_bytes"])
            trace = w["memory_samples"]
            assert trace
            assert (
                max(s["memory.swap.current"] for s in trace)
                == max(s["host_swap_used_bytes"] for s in trace)
                == 0
            )
            assert all(
                w["cgroup_events_before"][k] == w["cgroup_events_after"][k] == 0
                for k in ["oom", "oom_kill"]
            )
            rows.append(
                dict(
                    host=rank,
                    shape=shape,
                    parameters=count,
                    median_update_seconds=w["median_update_seconds"],
                    global_losses=[s["global_loss"] for s in w["samples"]],
                    measured_peak_live_MiB=max(
                        s["peak_allocated_bytes"] for s in w["samples"]
                    )
                    / 1024**2,
                    measured_peak_reserved_MiB=max(
                        s["peak_reserved_bytes"] for s in w["samples"]
                    )
                    / 1024**2,
                    cgroup_peak_MiB=max(s["memory.peak"] for s in trace) / 1024**2,
                    host_min_available_GiB=min(s["host_available_bytes"] for s in trace)
                    / GIB,
                    guardian_min_available_GiB=min(
                        s["available_bytes"] for s in f["headroom_samples"]
                    )
                    / GIB,
                    traffic_input_bytes=w["samples"][0]["traffic_input_bytes"],
                    worker_seconds=w["worker_seconds"],
                    guarded_phase_seconds=f["guarded_phase_seconds"],
                )
            )
            workers[label] = w
        assert parameters(CONFIG) * 16 > h["host_MemTotal_bytes"]
        return h, workers, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[3]
    )
    for name in ["host104", "host105", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prior = prerequisite(a.workspace)
    lh, lw, lr = read(a.host104, 0, prior)
    rh, rw, rr = read(a.host105, 1, prior)
    for label in LABELS:
        assert lw[label]["initial_global_loss"] == rw[label]["initial_global_loss"]
        assert [s["global_loss"] for s in lw[label]["samples"]] == [
            s["global_loss"] for s in rw[label]["samples"]
        ]
        assert [s["seconds_max_rank"] for s in lw[label]["samples"]] == [
            s["seconds_max_rank"] for s in rw[label]["samples"]
        ]
        assert (
            lw[label]["state"]["local_master_elements"]
            == rw[label]["state"]["local_master_elements"]
        )
    result = dict(
        status="passed",
        groups=24,
        guards=4,
        local_tests=112,
        rows=lr + rr,
        fresh_training_only_1B_capacity_qualified=True,
        long_run_or_restart_at_1B_qualified=False,
        performance_speedup_claim=False,
        native_runtime_changed=False,
        boot_ids=[lh["boot_id"], rh["boot_id"]],
        archives=[
            dict(
                path=str(p),
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
            for p in [a.host104, a.host105]
        ],
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
