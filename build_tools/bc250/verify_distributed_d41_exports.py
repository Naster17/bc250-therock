#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D41 CPU oracle publication and exact qualified control without raw exports."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from capacity_plan_d41 import CONFIG, CONTROL, PARAMETERS, admit, headroom
from verify_distributed_d35_exports import read as read_d35

# Frozen at implementation; never follow mutable sources during historical replay.
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
    "cpu_checkpoint_policy_d38.py": "66b9eab15acf3e10ae167c7a75fa05c77d9d231f8b7a4c32425216a9da7908f2",
    "stream_contract_d37.py": "4d54498bf78e21c2fefa256c296c881948c07b7c5f68a3ff4cbc84ba738ce51f",
    "capacity_plan_d41.py": "948aabff72f7534a1231749b67766e4040f9b6a459c0eed6d7777b44b0fede25",
    "fixture_d41.py": "ca54426e99b0ae26763177d616942f7796aaead2013ff69098f30e2b9bc2e9b5",
    "artifact_d41.py": "c1a3b54851f68b3ed3c586bf57d844d8ccfd936b60b96c06039f17525cdb0dd6",
    "cpu_oracle_d41.py": "cd1f9df577dde6ff2c238f665c614c10dad2d4366b591b709849eb9192aa9434",
}
HOST_SHA256 = {
    "guard_distributed_d41_on_host.py": "431b720b4c00166ea6b322596313ecad46d668548c5e4ee8c22c31a08023c260",
    "cpu_d41_bounded_launch.py": "ca316c49a61e0a129221aad728efabf0ef100bb30a7fb2e5a16128e9d88fc4de",
    "export_distributed_d41_on_host.py": "79c0aab9b699259581039353475f0e2cfc8d59c7cd944106144292ed350ed5ec",
}
CHECKPOINT_SHA = "eab49639231cc989d65c04212923eef0ade4aa059b68bcabfe2819007f34e69f"


def read(path, baseline, baseline_path):
    prefix = "home/nik/bc250work/evidence/distributed-d41-r1/"
    source = "home/nik/bc250work/tools/distributed-d41/"
    with tarfile.open(baseline_path) as old:
        old_manifest = json.load(
            old.extractfile(
                "home/nik/bc250work/evidence/distributed-d35-r1/cpu-reference/reference/manifest.json"
            )
        )
    with tarfile.open(path) as archive:

        def blob(name):
            return archive.extractfile(prefix + name).read()

        def record(name):
            return json.loads(blob(name))

        h = record("handoff.json")
        assert (
            h["status"] == "passed"
            and h["original_D35_reference_blobs_reverified"]
            and h["large_reference_blobs_reverified"]
        )
        assert (
            h["GPU_admitted"] is False
            and h["large_reference_payload_bytes"] == PARAMETERS * 12
        )
        assert h["boot_id"] == baseline["boot_id"] and h["kernel"] == baseline["kernel"]
        assert record("CPU-control-manifest.json") == old_manifest
        assert (
            record("CPU-control-worker.json")
            == baseline["records"]["cpu-reference"]["worker"]
        )
        assert (
            record("CPU-control-final.json")
            == baseline["records"]["cpu-reference"]["final"]
        )
        assert not any(
            m.name.endswith(".raw") or "roundtrip-scratch" in m.name
            for m in archive.getmembers()
        )
        for filename, wanted in {**SOURCE_SHA256, **HOST_SHA256}.items():
            assert (
                hashlib.sha256(
                    archive.extractfile(source + filename).read()
                ).hexdigest()
                == wanted
            )
        large_manifest = record("large-oracle/reference/manifest.json")
        assert large_manifest["schema"] == 1 and large_manifest["status"] == "passed"
        assert (
            large_manifest["config"] == CONFIG
            and large_manifest["parameters"] == PARAMETERS
        )
        assert (
            large_manifest["sources"] == SOURCE_SHA256
            and large_manifest["cuda_initialized"] is False
        )
        names, files, elements = set(), set(), 0
        import math

        for entry in large_manifest["tensors"]:
            assert entry["name"] not in names and all(
                type(d) is int and d > 0 for d in entry["shape"]
            )
            names.add(entry["name"])
            count = math.prod(entry["shape"])
            elements += count
            for key, width in [("initial", 4), ("gradient", 8)]:
                item = entry[key]
                assert Path(item["file"]).name == item["file"] and item[
                    "file"
                ].endswith(".raw")
                assert item["file"] not in files and item["bytes"] == count * width
                assert len(item["sha256"]) == 64 and set(item["sha256"]) <= set(
                    "0123456789abcdef"
                )
                files.add(item["file"])
        assert len(names) == 244 and len(files) == 488 and elements == PARAMETERS
        assert math.isfinite(large_manifest["CPU_FP64_loss"])
        rows, workers, uuids, cids = [], {}, set(), set()
        for label, shape, config in [
            ("control", "control", CONTROL),
            ("large-oracle", "large", CONFIG),
        ]:
            w, f = record(label + "/rank.json"), record(label + "/final.json")
            assert h["records"][label] == dict(worker=w, final=f) and f["worker"] == w
            assert (
                f["status"] == w["status"] == "passed"
                and w["shape"] == f["shape"] == shape
            )
            assert w["config"] == config and w["sources"] == SOURCE_SHA256
            assert (
                w["physical_rank"] == f["rank"] == 0 and w["CUDA_initialized"] is False
            )
            assert len(w["checks"]) == 8 and "headroom_abort" not in f
            assert w["birth"]["uuid"] not in uuids and f["container_id"] not in cids
            uuids.add(w["birth"]["uuid"])
            cids.add(f["container_id"])
            assert (
                f["kernel_log_unchanged"]
                and f["registry_unchanged"]
                and f["agent_shutdown_clean"]
            )
            assert (
                f["registry_sha256"]
                == baseline["records"]["cpu-reference"]["final"]["registry_sha256"]
            )
            assert f["docker_containers"] == dict(before=[], after=[])
            assert blob(label + "/kernel-before.log") == blob(
                label + "/kernel-after.log"
            )
            state, ack = record(label + "/guard/state.json"), record(
                label + "/guard/ack.json"
            )
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "boot_id", "phase"]
            )
            assert state["boot_id"] == f["boot_id"] == h["boot_id"]
            for plan in [w["plan"], f["admission"]]:
                assert plan == admit(
                    shape, plan["available_before_bytes"], plan["free_before_bytes"]
                )
            assert w["parameters"] == w["gradient_elements"] == w["plan"]["parameters"]
            assert f["headroom_samples"]
            for sample in f["headroom_samples"]:
                headroom(
                    sample["memory_bytes"],
                    sample["available_bytes"],
                    sample["free_bytes"],
                )
            assert w["checkpoint_options"] == dict(
                use_reentrant=False, preserve_rng_state=True
            )
            assert all(type(v) is bool for v in w["checkpoint_options"].values())
            assert w["checkpoint_source_sha256"] == CHECKPOINT_SHA
            assert w["checkpoint_runtime_modules"] == config["blocks"] + 1
            assert w["block_calls_after_forward"] == {
                str(i): 1 for i in range(config["blocks"])
            }
            assert w["block_calls_after_backward"] == {
                str(i): 2 for i in range(config["blocks"])
            }
            assert w["CPU_rng_before_sha256"] == w["CPU_rng_after_sha256"]
            tensors = (
                old_manifest["tensors"]
                if shape == "control"
                else large_manifest["tensors"]
            )
            assert w["initial_sha256"] == {
                e["name"]: e["initial"]["sha256"] for e in tensors
            }
            assert w["gradient_sha256"] == {
                e["name"]: e["gradient"]["sha256"] for e in tensors
            }
            assert w["gradient_parameters"] == len(tensors) and set(
                w["hook_counts"]
            ) == set(w["gradient_sha256"])
            assert all(type(v) is int and v == 1 for v in w["hook_counts"].values())
            assert w["retained_gradient_elements_after_backward"] == 0
            assert (
                w["maximum_visible_gradient_elements_at_hook"]
                <= w["plan"]["largest_FP64_tensor_bytes"] // 8
            )
            assert (
                w["maximum_roundtrip_bytes"] == w["plan"]["largest_FP64_tensor_bytes"]
            )
            assert (
                w["CPU_FP64_loss"]
                == (old_manifest if shape == "control" else large_manifest)[
                    "CPU_FP64_loss"
                ]
            )
            if shape == "control":
                assert (
                    f["CPU_artifact_manifest_sha256"]
                    == baseline["records"]["cpu-reference"]["worker"]["manifest_sha256"]
                )
            else:
                assert (
                    f["control_proof_sha256"]
                    == hashlib.sha256(blob("control/final.json")).hexdigest()
                )
                assert (
                    w["manifest_sha256"]
                    == hashlib.sha256(
                        blob("large-oracle/reference/manifest.json")
                    ).hexdigest()
                )
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
                    shape=shape,
                    parameters=w["parameters"],
                    CPU_FP64_loss=w["CPU_FP64_loss"],
                    whole_cgroup_peak_MiB=max(s["memory.peak"] for s in trace)
                    / 1024**2,
                    process_peak_RSS_MiB=max(
                        s["process_RSS_high_water_bytes"] for s in trace
                    )
                    / 1024**2,
                    host_min_available_GiB=min(s["host_available_bytes"] for s in trace)
                    / 1024**3,
                    final_free_disk_GiB=w["snapshots"][-1]["free_disk_bytes"] / 1024**3,
                    retained_gradient_elements=0,
                    maximum_roundtrip_bytes=w["maximum_roundtrip_bytes"],
                    worker_seconds=w["worker_seconds"],
                    guarded_phase_seconds=f["guarded_phase_seconds"],
                )
            )
            workers[label] = w
        return h, workers, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["host104", "baseline104", "baseline-results", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prior = json.loads(a.baseline_results.read_text())
    assert (
        prior["status"] == "passed"
        and hashlib.sha256(a.baseline104.read_bytes()).hexdigest()
        == prior["archives"][0]["sha256"]
    )
    baseline, _ = read_d35(a.baseline104, 0)
    h, workers, rows = read(a.host104, baseline, a.baseline104)
    result = dict(
        status="passed",
        groups=16,
        guards=2,
        local_tests=97,
        host="104",
        boot_id=h["boot_id"],
        CPU_large_oracle_published=True,
        GPU_admitted=False,
        larger_GPU_fit_qualified=False,
        native_runtime_changed=False,
        reference_payload_bytes=h["large_reference_payload_bytes"],
        manifest_sha256=workers["large-oracle"]["manifest_sha256"],
        control_matches_D35_exactly=True,
        rows=rows,
        baseline_archive=prior["archives"][0],
        archive=dict(
            path=str(a.host104),
            bytes=a.host104.stat().st_size,
            sha256=hashlib.sha256(a.host104.read_bytes()).hexdigest(),
        ),
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
