#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay D42 oracle/readiness/resource proofs and ALL native audit references."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from uuid import UUID
from verify_distributed_d35_exports import read as read35
from verify_distributed_d36_exports import read as read36
from verify_distributed_d39_exports import read as read39
from verify_distributed_d40_exports import read as read40
from verify_distributed_d41_exports import read as read41
from capacity_plan_d41 import CONFIG, CONTROL, parameters, GIB
from capacity_plan_d42 import (
    admit,
    headroom,
    HOST_REGISTRY_SHA256,
    READINESS_SECONDS,
    GUARD_SECONDS,
)
from unit_plan_d36 import partition

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
    "cpu_checkpoint_policy_d38.py": "66b9eab15acf3e10ae167c7a75fa05c77d9d231f8b7a4c32425216a9da7908f2",
    "stream_contract_d37.py": "4d54498bf78e21c2fefa256c296c881948c07b7c5f68a3ff4cbc84ba738ce51f",
    "capacity_plan_d41.py": "948aabff72f7534a1231749b67766e4040f9b6a459c0eed6d7777b44b0fede25",
    "fixture_d41.py": "ca54426e99b0ae26763177d616942f7796aaead2013ff69098f30e2b9bc2e9b5",
    "artifact_d41.py": "c1a3b54851f68b3ed3c586bf57d844d8ccfd936b60b96c06039f17525cdb0dd6",
    "cpu_oracle_d41.py": "cd1f9df577dde6ff2c238f665c614c10dad2d4366b591b709849eb9192aa9434",
    "reference_transport_d40.py": "829043e2aab1877f2ac3805c7d4264df7b22db92aa04615ad02556a5207726bf",
    "capacity_plan_d42.py": "fbd64603efefaef31e93d4214e0774e43407a87950d5567218afd3b6d54b9e4b",
    "distributed_d42_capacity.py": "de185f0a3d5306ce113e02e8d21b2f43df360fb0393bdaaac55b40ae763cb5b7",
    "serve_reference_d42.py": "8300b5e18a69110e7a2378b32cc22962ab81c539f8644444c3124edca634627b",
    "oracle_descriptor_d42.json": "a0af0b1f62295742d23b8e809ac320e3da60d0d8fdc2e4783d9fa6c1e068a867",
    "audit_chunks_d42.py": "d1a5b2b169b75124d487075689cc07848083ef617505028468f90d8c2429f441",
    "cpu_adamw_chunks_d42.py": "9e9d7e5ede0facad96a6339530ecb160c33b5800e024b6415f5e542f94a99cd4",
}
HOST_SHA256 = {
    "guard_distributed_d42_on_host.py": "19149e5121cfa3b634f2db49c9d223f230e76dfa36b23070cc2b2ee2b5bd33f6",
    "distributed_d42_bounded_launch.py": "574372d65a3f7023c70a7fdce208d1ef2d74cfb525545f2c965442695c99b148",
    "deadline_guard_d42.py": "bce97bfb137c77ff28f1ad30ea23f9aada593174c58e91a2e1628a5b8b011d7c",
    "export_distributed_d42_on_host.py": "00904838a268802a25d8c7c256469e8b1f7043069c00f7b1e24ae16268f1796c",
}
LABELS = ["control", "large-audit"]


def predecessors(workspace):
    reports = workspace / "therock/docs/development"
    previous = {}

    def published(task):
        return json.loads(
            (
                reports / f"bc250_distributed_d{task}_results/verified-results.json"
            ).read_text()
        )

    def archive(item):
        path = workspace / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        return path

    pubs = {task: published(task) for task in [35, 36, 39, 40, 41]}
    for rank in range(2):
        p35 = archive(pubs[35]["archives"][rank])
        h35, _ = read35(p35, rank)
        _, w36, _ = read36(archive(pubs[36]["archives"][rank]), rank, h35, p35)
        _, w39, _ = read39(archive(pubs[39]["archives"][rank]), rank, h35, p35, w36)
        h40, w40, _ = read40(
            archive(pubs[40]["archives"][rank]), rank, h35, p35, w36, w39
        )
        previous[rank] = (h40, w40)
    p41 = archive(pubs[41]["archive"])
    h41, _, _ = read41(
        p41,
        read35(archive(pubs[35]["archives"][0]), 0)[0],
        archive(pubs[35]["archives"][0]),
    )
    return previous, h41


def read(path, rank, old, oracle):
    prefix = "home/nik/bc250work/evidence/distributed-d42-r5/"
    source = "home/nik/bc250work/tools/distributed-d42/"
    with tarfile.open(path) as archive:

        def blob(name):
            return archive.extractfile(prefix + name).read()

        def record(name):
            return json.loads(blob(name))

        h = record("handoff.json")
        assert h["status"] == "passed" and h["rank"] == rank
        # The export may follow a later reboot. Each test and its ACK must still
        # belong to the qualified test boot; never substitute the export boot.
        assert str(UUID(h["boot_id"])) == h["boot_id"]
        assert h["kernel"] == old[0]["kernel"]
        assert h["original_source_blobs_reverified_on_104"] == (rank == 0)
        assert not any(m.name.endswith(".raw") for m in archive.getmembers())
        for filename, pin in {**SOURCE_SHA256, **HOST_SHA256}.items():
            assert (
                hashlib.sha256(
                    archive.extractfile(source + filename).read()
                ).hexdigest()
                == pin
            )
        descriptor = json.load(
            archive.extractfile(source + "oracle_descriptor_d42.json")
        )
        assert (
            descriptor["large"]["worker"] == oracle["records"]["large-oracle"]["worker"]
        )
        assert (
            descriptor["large"]["final"] == oracle["records"]["large-oracle"]["final"]
        )
        rows = []
        workers = {}
        uuids = set()
        cids = set()
        for label, shape, config in zip(
            LABELS, ["control", "large"], [CONTROL, CONFIG]
        ):
            w, f = record(label + "/rank.json"), record(label + "/final.json")
            assert h["records"][label] == dict(worker=w, final=f) and f["worker"] == w
            assert (
                w["status"] == f["status"] == "passed"
                and w["shape"] == f["shape"] == shape
            )
            assert (
                w["config"] == config
                and w["mode"] == f["mode"] == "fsdp"
                and w["phase"] == "qualify"
            )
            assert (
                w["rank"] == w["physical_rank"] == f["rank"] == rank and w["world"] == 2
            )
            assert (
                len(w["checks"]) == 6
                and w["CPU_oracle_executed_in_GPU_worker"] is False
            )
            assert (
                w["CPU_reference_executed_in_this_worker"] is False
                and w["full_CPU_optimizer_or_export"] is False
            )
            assert w["sources"] == SOURCE_SHA256
            assert (
                f["kernel_log_unchanged"]
                and f["registry_unchanged"]
                and f["agent_shutdown_clean"]
            )
            assert f["registry_sha256"] == HOST_REGISTRY_SHA256[rank] and f[
                "docker_containers"
            ] == dict(before=[], after=[])
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
            assert all(state[k] == ack[k] for k in ["nonce", "boot_id", "phase"])
            assert state["boot_id"] == old[0]["boot_id"] == f["boot_id"]
            assert w["birth"]["uuid"] not in uuids and f["container_id"] not in cids
            uuids.add(w["birth"]["uuid"])
            cids.add(f["container_id"])
            manifest = descriptor[shape]["manifest"]
            count = parameters(config)
            entries = {e["name"]: e for e in manifest["tensors"]}
            pin = descriptor[shape]["manifest_sha256"]
            assert (
                w["CPU_artifact_manifest_sha256"]
                == f["CPU_artifact_manifest_sha256"]
                == pin
            )
            ready = w["readiness"]
            identity = dict(
                manifest_sha256=pin,
                shape=shape,
                sources=SOURCE_SHA256,
                CUDA_initialized=False,
            )
            assert ready["peers"] == [identity, identity] and ready["CPU_only"] is True
            assert (
                ready["timeout_seconds"] == READINESS_SECONDS
                and ready["native_GPU_PG_timeout_seconds"] == 120
            )
            assert 0 <= ready["verification_and_wait_seconds"] < READINESS_SECONDS + 180
            assert ready["port"] == (29744 if shape == "control" else 29746)
            for plan in [f["admission"], w["plan"]]:
                assert plan == admit(
                    shape,
                    plan["available_before_bytes"],
                    plan["free_disk_before_bytes"],
                    device_free=plan["device_free_before_bytes"],
                )
            assert "headroom_abort" not in f and f["headroom_samples"]
            for s in f["headroom_samples"]:
                headroom(s["memory_bytes"], s["available_bytes"], s["free_bytes"])
            assert w["state"]["full_parameter_elements"] == count
            assert (
                w["state"]["local_master_elements"]
                == w["state"]["AdamW_moment_elements_each"]
                == count // 2
            )
            ref = w["reference"]
            assert (
                ref["gradient_parameters"] == len(entries)
                and ref["gradient_elements"] == count
            )
            assert ref["CPU_FP64_loss"] == manifest["CPU_FP64_loss"] and math.isfinite(
                ref["loss"]
            )
            assert ref["ALL_AdamW_parameters_moments_counters"] is True
            assert (
                ref["actual_raw_spool_bytes"] == 0
                and ref["actual_raw_spool_removed"] is True
            )
            assert (
                ref["clipping_audit"] == "independent-CPU-FP64-reference"
                and ref["clipping_atol"] == 1e-8
                and ref["clipping_rtol"] == 5e-5
            )
            assert ref["local_moment_elements_each"] == count // 2
            assert ref["maximum_CPU_moment_elements"] == 2 * 1024**2
            assert ref["CPU_AdamW_chunk_elements"] == 1024**2
            assert ref["CPU_AdamW_chunks_checked"] == sum(
                (math.prod(e["shape"]) + 1024**2 - 1) // 1024**2
                for e in entries.values()
            )
            assert ref["CPU_AdamW_values_checked"] == count
            assert ref["CPU_AdamW_chunk_equivalence"] == dict(
                chunk_elements=1024**2,
                synthetic_elements=2 * 1024**2 + 13,
                chunks=3,
                ALL_values_moments_step_exact=True,
            )
            for key in [
                "actual_raw_spool_sha256",
                "clipped_gradient_sha256",
                "updated_parameter_sha256",
            ]:
                assert set(ref[key]) == set(entries) and all(
                    len(v) == 64 for v in ref[key].values()
                )
            if shape == "control":
                for key, value in old[1]["remote-reference"]["reference"].items():
                    if key not in {"audit", "maximum_CPU_moment_elements"}:
                        assert ref[key] == value, (key, "D40")
            cache = ref["cache_trim_events"]
            assert [e["stage"] for e in cache] == [
                "after-wrap-before-backward",
                "raw-gradient",
                "clipped-gradient",
                "updated-parameter",
                "before-local-moments",
            ]
            assert all(
                e["after"]["reserved_bytes"] <= e["before"]["reserved_bytes"]
                for e in cache
            )
            audit = ref["audit"]
            assert (
                audit["kind"] == "unit"
                and audit["whole_model_summons_forbidden"] is True
            )
            assert audit["ownership"] == partition(
                entries, [""] + [f"transformer.h.{i}" for i in range(config["blocks"])]
            )
            assert len(audit["events"]) == 3 * (config["blocks"] + 1)
            for stage in ["raw-gradient", "clipped-gradient", "updated-parameter"]:
                events = [e for e in audit["events"] if e["stage"] == stage]
                names = [n for e in events for n in e["parameter_names"]]
                assert len(events) == config["blocks"] + 1 and len(names) == len(
                    set(names)
                ) == len(entries)
                assert set(names) == set(entries) and all(
                    e["recurse"] is False for e in events
                )
            assert (
                max(e["full_parameter_elements"] for e in audit["events"])
                == 12 * config["width"] ** 2 + 13 * config["width"]
            )
            payload = count * 12
            files = len(entries) * 2
            transport = w["reference_transport"]
            assert transport["kind"] == ("local" if rank == 0 else "remote")
            if rank == 1:
                for stats, reads in [
                    (transport["stats"], files * 2),
                    (f["remote_preflight"], 0),
                ]:
                    assert (
                        stats["endpoint"] == "http://192.168.1.104:29842"
                        and stats["manifest_sha256"] == pin
                    )
                    assert (
                        stats["verified_blobs"] == files
                        and stats["verified_bytes"] == payload
                    )
                    assert stats["read_blobs"] == reads and stats["read_bytes"] == (
                        2 * payload if reads else 0
                    )
                    assert (
                        stats["disk_cache_bytes"] == 0
                        and stats["local_fallback_used"] is False
                    )
                    assert stats["max_retained_blob_bytes"] == (
                        max(e["gradient"]["bytes"] for e in entries.values())
                        if reads
                        else 0
                    )
                assert (
                    json.loads(blob(label + "/remote-reference/manifest.json"))
                    == manifest
                )
            else:
                assert transport["stats"] is None
                server = record(label + "/source.json")
                assert (
                    f["reference_source"] == server
                    and server["status"] == "stopped"
                    and server["uid"] == 1000
                )
                assert (
                    server["manifest_sha256"] == pin
                    and server["source_blobs_reverified"] == files
                )
                stats = server["transport"]
                assert (
                    stats["completed_requests"] == files * 4 + 1
                    and stats["completed_bytes"]
                    == payload * 4 + descriptor[shape]["manifest_bytes"]
                )
                assert stats["blob_counts"] == {
                    "manifest.json": 1,
                    **{
                        e[k]["file"]: 4
                        for e in entries.values()
                        for k in ["initial", "gradient"]
                    },
                }
                assert stats["failed_requests"] == stats["rejected_requests"] == 0
            trace = w["memory_samples"]
            assert trace and all(
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
                    shape=shape,
                    loss=ref["loss"],
                    CPU_loss=ref["CPU_FP64_loss"],
                    max_raw_gradient_error=ref["maximum_absolute_gradient_error"],
                    max_clip_error=ref["maximum_absolute_CPU_reference_clip_error"],
                    local_state_elements=count // 2,
                    peak_live_audit_MiB=audit["peak_allocated_bytes"] / 1024**2,
                    peak_reserved_audit_MiB=audit["peak_reserved_bytes"] / 1024**2,
                    cgroup_peak_MiB=max(s["memory.peak"] for s in trace) / 1024**2,
                    host_min_available_GiB=min(s["host_available_bytes"] for s in trace)
                    / GIB,
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
    for n in ["host104", "host105", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    prior, oracle = predecessors(a.workspace)
    left, lw, lr = read(a.host104, 0, prior[0], oracle)
    right, rw, rr = read(a.host105, 1, prior[1], oracle)
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
        local_tests=108,
        parameters=parameters(CONFIG),
        source_bound_GPU_oracle_audit_qualified=True,
        single_resident_FP32_state_floor_exceeds_host_RAM=True,
        training_only_capacity_or_speed_qualified=False,
        native_runtime_changed=False,
        PLAN05_complete=False,
        rows=lr + rr,
        test_boot_ids=[prior[0][0]["boot_id"], prior[1][0]["boot_id"]],
        export_boot_ids=[left["boot_id"], right["boot_id"]],
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
