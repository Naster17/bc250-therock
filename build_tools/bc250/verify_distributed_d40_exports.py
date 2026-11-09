#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify bounded D40 remote reads and exact D39/D36 native GPU audit state."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from capacity_plan_d35 import CONFIG, PARAMETERS, GIB
from verify_distributed_d35_exports import read as read_d35
from verify_distributed_d36_exports import read as read_d36
from verify_distributed_d39_exports import read as read_d39
from artifact_d35 import SOURCES as ORACLE_FILES
from unit_plan_d36 import partition

LABELS = ["local-control", "remote-reference"]
# Pinned below after implementation; historical verification never follows mutable files.
SOURCE_SHA256 = {
    "artifact_d35.py": "9b60d0750ec09185a5d0a7abaa2f51f4ee4940609003971e0761306528876689",
    "capacity_plan_d35.py": "cf481a6b480ed16b9ecc831abc72a044a19d5fba57ed65d76d245a54b98d741e",
    "fixture_d35.py": "4c0c799732d5721679e3a77038497ffffe6d92e83b8e873d67840a03d5a57ad3",
    "prepare_distributed_d35_reference.py": "ab576ea3bbebdbf2657abc77f4d812239c2f6e8497ae9ee0d21761c210d28d6a",
    "distributed_d35_growth.py": "a4b78d1fdcc1203cb65bf8257dde22b1ff284b631abae5a01982d0eacbdaca90",
    "distributed_d32_capacity.py": "f6e78720483d065c55a55e37df6cc604dd8bfc2df5e4f408ae6144b32af225ab",
    "distributed_d33_fresh.py": "72ca07b27d9b23d84865f05505c2d6212d401bdace9acdc9df574252ad64b1d4",
    "capacity_plan_d32.py": "8c7687e47917d79c6433322f296493a98ea7bcd555333a0a27bdbae4985f4bbc",
    "qualification_d33.py": "7b20334af874b9cbaa656a28118639719bc87c3a88c0521682a02f4b0f5f3a10",
    "distributed_d05_train.py": "a598b26b0c30ddd1a6efe205a4e28bce8a03f1f69467aa0bdf82fbe7c7e750bb",
    "distributed_d36_units.py": "15caa9f773b0bd97762d9b44d3a9c496b9448edf434600ec7bd2a522b8f72c1c",
    "audit_units_d36.py": "bf70292465f6664365f15a9721f4db69f36dd2153b8bd1846bf82eb01c9c3f65",
    "unit_plan_d36.py": "ea2b434a6803eeabb37561aa9eb2eb2929fa8c21ebde2094bfceb15f00f944c2",
    "distributed_d39_units.py": "7666b233bee93e6de3d092e9e59ccc029d32f090316b077a7091634f357ba544",
    "audit_units_d39.py": "91a8c139437bd7abbd8263cfc2dc48b2d2575a50063fdf62d1f70d035165de9e",
    "reference_transport_d40.py": "829043e2aab1877f2ac3805c7d4264df7b22db92aa04615ad02556a5207726bf",
    "distributed_d40_transport.py": "d5a20eb2ca367c11fa9311245a175308903c00f33e138f11ecdd3afbbe3a0310",
    "serve_reference_d40.py": "128b4241f20037ac12210efad5ce4e473fab50429c93dd0718aa75232b714112",
}


def read(path, rank, baseline, baseline_path, legacy_workers, previous_workers):
    prefix = "home/nik/bc250work/evidence/distributed-d40-r1/"
    source = "home/nik/bc250work/tools/distributed-d40/"
    with tarfile.open(baseline_path) as old_archive:
        old_manifest = json.load(
            old_archive.extractfile(
                "home/nik/bc250work/evidence/distributed-d35-r1/cpu-reference/reference/manifest.json"
            )
        )
    with tarfile.open(path) as archive:

        def blob(name):
            return archive.extractfile(prefix + name).read()

        def record(name):
            return json.loads(blob(name))

        h = record("handoff.json")
        assert h["status"] == "passed" and h["final_reference_blobs_reverified"] is True
        assert record("CPU-reference-manifest.json") == old_manifest
        cpu = record("CPU-reference-worker.json")
        prior = record("CPU-reference-final.json")
        assert (
            cpu == baseline["records"]["cpu-reference"]["worker"]
            and prior == baseline["records"]["cpu-reference"]["final"]
        )
        assert (
            hashlib.sha256(blob("CPU-reference-manifest.json")).hexdigest()
            == cpu["manifest_sha256"]
        )
        assert old_manifest["sources"] == {n: SOURCE_SHA256[n] for n in ORACLE_FILES}
        entries = {entry["name"]: entry for entry in old_manifest["tensors"]}
        assert len(entries) == 148
        assert not any(m.name.endswith(".raw") for m in archive.getmembers())
        original_cids = {
            rec["final"]["container_id"] for rec in baseline["records"].values()
        }
        uuids, cids = set(), set()
        rows = []
        workers = {}
        for label in LABELS:
            w, f = record(label + "/rank.json"), record(label + "/final.json")
            assert h["records"][label] == dict(worker=w, final=f) and f["worker"] == w
            assert w["status"] == f["status"] == "passed"
            assert (
                w["config"] == CONFIG
                and w["mode"] == f["mode"] == "fsdp"
                and w["phase"] == f["phase"] == "qualify"
            )
            assert w["physical_rank"] == w["rank"] == rank and w["world"] == 2
            assert (
                len(w["checks"]) == 6
                and w["CPU_oracle_executed_in_GPU_worker"] is False
            )
            assert (
                not w["CPU_reference_executed_in_this_worker"]
                and not w["full_CPU_optimizer_or_export"]
            )
            assert (
                w["CPU_artifact_manifest_sha256"]
                == f["CPU_artifact_manifest_sha256"]
                == cpu["manifest_sha256"]
            )
            assert (
                f["registry_sha256"]
                == baseline["records"]["qualify-fsdp"]["final"]["registry_sha256"]
            )
            assert (
                f["registry_unchanged"]
                and f["kernel_log_unchanged"]
                and f["agent_shutdown_clean"]
            )
            assert f["docker_containers"] == dict(before=[], after=[])
            assert blob(label + "/kernel-before.log") == blob(
                label + "/kernel-after.log"
            )
            state, ack = record(label + "/guard/state.json"), record(
                label + "/guard/ack.json"
            )
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            assert state["boot_id"] == h["boot_id"] == baseline["boot_id"]
            assert (
                w["birth"]["uuid"] not in uuids
                and f["container_id"] not in cids | original_cids
            )
            uuids.add(w["birth"]["uuid"])
            cids.add(f["container_id"])
            assert w["sources"] == SOURCE_SHA256
            assert f["boot_id"] == state["boot_id"]
            assert f["kernel"] == h["kernel"] == baseline["kernel"]
            for filename, wanted in SOURCE_SHA256.items():
                assert (
                    hashlib.sha256(
                        archive.extractfile(source + filename).read()
                    ).hexdigest()
                    == wanted
                )
            assert w["state"]["full_parameter_elements"] == PARAMETERS
            assert (
                w["state"]["local_master_elements"]
                == w["state"]["AdamW_moment_elements_each"]
                == PARAMETERS // 2
            )
            ref = w["reference"]
            original = baseline["records"]["qualify-fsdp"]["worker"]["reference"]
            for key, wanted in original.items():
                # D36 sums checked per-parameter squares in CPU manifest order.
                # D35's FSDP named-parameter order differs; these two report-only
                # scalars can differ by a double-precision rounding unit. The
                # actual GPU/CPU loss, gradient, norm, clipping and AdamW gates
                # are unchanged in both workers; raw-gradient hashes stay exact.
                if key == "CPU_root_norm":
                    assert abs(ref[key] - wanted) <= 1e-13
                elif key == "clip_coefficient":
                    assert abs(ref[key] - wanted) <= 1e-15
                    assert ref[key] == min(1.0, 0.5 / (ref["CPU_root_norm"] + 1e-6))
                elif key == "actual_raw_spool_bytes":
                    assert ref[key] == 0
                else:
                    assert ref[key] == wanted, (label, key)
            prior_ref = legacy_workers["unit-audit"]["reference"]
            for key, wanted in prior_ref.items():
                if key not in {"audit", "actual_raw_spool_bytes"}:
                    assert ref[key] == wanted, (label, "D36", key)
            assert ref["clipping_audit"] == "independent-CPU-FP64-reference"
            assert ref["clipping_atol"] == 1e-8 and ref["clipping_rtol"] == 5e-5
            assert (
                w["checks"][2]
                == "CPU-root-norm-independent-FP64-reference-ALL-clipped-gradients"
            )
            for key, wanted in previous_workers["reference-clip"]["reference"].items():
                if key != "audit":
                    assert ref[key] == wanted, (label, "D39", key)
            transport = w["reference_transport"]
            remote = rank == 1 and label == "remote-reference"
            assert transport["kind"] == ("remote" if remote else "local")
            payload = sum(
                e[k]["bytes"] for e in entries.values() for k in ["initial", "gradient"]
            )
            largest = max(
                e[k]["bytes"] for e in entries.values() for k in ["initial", "gradient"]
            )
            if remote:
                for stats, reads in [
                    (transport["stats"], 592),
                    (f["remote_preflight"], 0),
                ]:
                    assert stats["endpoint"] == "http://192.168.1.104:29840"
                    assert stats["manifest_sha256"] == cpu["manifest_sha256"]
                    assert (
                        stats["verified_blobs"] == 296
                        and stats["verified_bytes"] == payload
                    )
                    assert stats["read_blobs"] == reads and stats["read_bytes"] == (
                        2 * payload if reads else 0
                    )
                    assert stats["max_retained_blob_bytes"] == (largest if reads else 0)
                    assert (
                        stats["verify_chunk_bytes"] == 1024**2
                        and stats["disk_cache_bytes"] == 0
                    )
                    assert stats["local_fallback_used"] is False
                assert blob(label + "/remote-reference/manifest.json") == blob(
                    "CPU-reference-manifest.json"
                )
                descriptor_members = [
                    m
                    for m in archive.getmembers()
                    if m.isfile()
                    and m.name.startswith(prefix + label + "/remote-reference/")
                ]
                assert len(descriptor_members) == 1
            else:
                assert transport["stats"] is None
            if rank == 0 and label == "remote-reference":
                source_record = record(label + "/source.json")
                assert f["reference_source"] == source_record
                assert source_record["pid"] == f["source_pid"] > 0
                assert source_record["endpoint"] == "http://192.168.1.104:29840"
                assert (
                    source_record["status"] == "stopped"
                    and source_record["uid"] == 1000
                )
                assert source_record["manifest_sha256"] == cpu["manifest_sha256"]
                assert (
                    source_record["sources"] == old_manifest["sources"]
                    and source_record["source_blobs_reverified"] == 296
                )
                assert source_record["allowed_clients"] == ["192.168.1.105"]
                assert (
                    source_record["address_space_limit_bytes"] == 256 * 1024**2
                    and source_record["fd_limit"] == 64
                )
                assert (
                    source_record["directory_listing"]
                    is source_record["writes_allowed"]
                    is False
                )
                counters = source_record["transport"]
                assert counters["completed_requests"] == 1185
                assert counters["completed_bytes"] == payload * 4 + len(
                    blob("CPU-reference-manifest.json")
                )
                assert counters["failed_requests"] == counters["rejected_requests"] == 0
                assert counters["max_chunk_bytes"] == 1024**2
                assert counters["blob_counts"] == {
                    "manifest.json": 1,
                    **{
                        e[k]["file"]: 4
                        for e in entries.values()
                        for k in ["initial", "gradient"]
                    },
                }
            for storage in [w["storage"], f["storage"]]:
                assert (
                    storage["required_payload_bytes"] == ref["actual_raw_spool_bytes"]
                )
                assert storage["minimum_free_after_bytes"] == 5 * GIB
                assert (
                    storage["free_before_bytes"]
                    >= storage["required_payload_bytes"] + 5 * GIB
                )
            assert not any("actual-gradients" in m.name for m in archive.getmembers())
            assert (
                set(ref["clipped_gradient_sha256"])
                == set(ref["updated_parameter_sha256"])
                == set(entries)
            )
            assert sum(len(v) for v in ref["local_AdamW_sha256"].values()) > 0
            audit = ref["audit"]
            assert audit["kind"] == "unit" and f["transport"] == (
                "local" if label == "local-control" else "remote"
            )
            assert audit["whole_model_summons_forbidden"] is True
            assert audit["ownership"] == partition(
                entries, [""] + [f"transformer.h.{i}" for i in range(CONFIG["blocks"])]
            )
            assert len(audit["events"]) == 39
            for stage in ["raw-gradient", "clipped-gradient", "updated-parameter"]:
                events = [e for e in audit["events"] if e["stage"] == stage]
                assert len(events) == 13
                names = [name for event in events for name in event["parameter_names"]]
                assert len(names) == len(set(names)) == 148 and set(names) == set(
                    entries
                )
                assert sum(e["full_parameter_elements"] for e in events) == PARAMETERS
                assert all(e["recurse"] == False for e in events)
            max_unit = max(e["full_parameter_elements"] for e in audit["events"])
            expected_block = 12 * CONFIG["width"] ** 2 + 13 * CONFIG["width"]
            assert max_unit == expected_block
            assert (
                max(e["allocated_inside_bytes"] for e in audit["events"])
                <= audit["peak_allocated_bytes"]
            )
            trace = w["memory_samples"]
            assert trace and min(s["host_available_bytes"] for s in trace) > 3 * GIB
            assert all(
                w["cgroup_events_before"][k] == w["cgroup_events_after"][k] == 0
                for k in ["oom", "oom_kill"]
            )
            assert (
                max(s["host_swap_used_bytes"] for s in trace)
                == max(s["memory.swap.current"] for s in trace)
                == 0
            )
            rows.append(
                dict(
                    host=rank,
                    label=label,
                    peak_live_audit_MiB=audit["peak_allocated_bytes"] / 1024**2,
                    peak_reserved_audit_MiB=audit["peak_reserved_bytes"] / 1024**2,
                    max_full_unit_elements=max_unit,
                    summon_contexts=len(audit["events"]),
                    whole_cgroup_peak_MiB=max(s["memory.peak"] for s in trace)
                    / 1024**2,
                    host_min_available_GiB=min(s["host_available_bytes"] for s in trace)
                    / GIB,
                    host_swap_max_bytes=max(s["host_swap_used_bytes"] for s in trace),
                    cgroup_swap_max_bytes=max(s["memory.swap.current"] for s in trace),
                    guarded_phase_seconds=f["guarded_phase_seconds"],
                    worker_seconds=w["worker_seconds"],
                )
            )
            workers[label] = w
        whole, unit = [workers[label]["reference"] for label in LABELS]
        for key in whole:
            if key not in {"audit", "actual_raw_spool_bytes", "clipping_audit"}:
                assert whole[key] == unit[key], key
        return h, workers, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "host104",
        "host105",
        "baseline104",
        "baseline105",
        "baseline-results",
        "legacy104",
        "legacy105",
        "legacy-results",
        "previous104",
        "previous105",
        "previous-results",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    published = json.loads(args.baseline_results.read_text())
    assert published["status"] == "passed" and published["groups"] == 72
    baselines = []
    for rank, path in enumerate([args.baseline104, args.baseline105]):
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == published["archives"][rank]["sha256"]
        )
        baseline, _ = read_d35(path, rank)
        baselines.append(baseline)
    legacy_published = json.loads(args.legacy_results.read_text())
    assert legacy_published["status"] == "passed" and legacy_published["groups"] == 24
    old_workers = []
    for rank, path in enumerate([args.legacy104, args.legacy105]):
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == legacy_published["archives"][rank]["sha256"]
        )
        _, workers, _ = read_d36(
            path, rank, baselines[rank], [args.baseline104, args.baseline105][rank]
        )
        old_workers.append(workers)
    previous_published = json.loads(args.previous_results.read_text())
    assert (
        previous_published["status"] == "passed" and previous_published["groups"] == 24
    )
    previous_workers = []
    for rank, path in enumerate([args.previous104, args.previous105]):
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == previous_published["archives"][rank]["sha256"]
        )
        _, workers, _ = read_d39(
            path,
            rank,
            baselines[rank],
            [args.baseline104, args.baseline105][rank],
            old_workers[rank],
        )
        previous_workers.append(workers)
    left, lw, lr = read(
        args.host104,
        0,
        baselines[0],
        args.baseline104,
        old_workers[0],
        previous_workers[0],
    )
    right, rw, rr = read(
        args.host105,
        1,
        baselines[1],
        args.baseline105,
        old_workers[1],
        previous_workers[1],
    )
    for label in LABELS:
        for key in [
            "actual_raw_spool_sha256",
            "clipped_gradient_sha256",
            "updated_parameter_sha256",
        ]:
            assert lw[label]["reference"][key] == rw[label]["reference"][key]
    result = dict(
        status="passed",
        groups=24,
        guards=4,
        local_tests=93,
        config=CONFIG,
        parameters=PARAMETERS,
        CPU_reference_reused_from_D35=True,
        fresh_CPU_oracle_executed=False,
        ALL_cpu_gates_unchanged=True,
        legacy_D36_native_digests_exact=True,
        previous_D39_native_digests_exact=True,
        remote_reference_transport_qualified=True,
        target_raw_reference_cache_bytes=0,
        target_max_retained_blob_bytes=100663296,
        source_payload_bytes_transferred=16941957120,
        source=left["records"]["remote-reference"]["final"]["reference_source"],
        remote_preflight=right["records"]["remote-reference"]["final"][
            "remote_preflight"
        ],
        remote_worker=rw["remote-reference"]["reference_transport"],
        independent_CPU_reference_clipping_qualified=True,
        temporary_raw_spool_bytes_before=0,
        temporary_raw_spool_bytes_after=0,
        paired_digests_exact=True,
        per_unit_scope_qualified=True,
        native_runtime_changed=False,
        training_speed_improvement_qualified=False,
        larger_than_single_fit_qualified=False,
        PLAN05_complete=False,
        rows=lr + rr,
        boot_ids=[left["boot_id"], right["boot_id"]],
        baseline_archives=published["archives"],
        legacy_archives=legacy_published["archives"],
        previous_archives=previous_published["archives"],
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
