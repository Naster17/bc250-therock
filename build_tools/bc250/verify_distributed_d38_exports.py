#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify exact CPU streaming against original D35 reference and a fresh control."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from capacity_plan_d35 import CONFIG, PARAMETERS, GIB
from artifact_d35 import SOURCES as ORACLE_FILES
from verify_distributed_d35_exports import read as read_d35
from verify_distributed_d37_exports import read as read_d37
from cpu_checkpoint_policy_d38 import validate_options

LABELS = ["stream-control", "checkpoint-stream"]
CHECKPOINT_SOURCE_SHA256 = (
    "eab49639231cc989d65c04212923eef0ade4aa059b68bcabfe2819007f34e69f"
)
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
    "cpu_checkpoint_stream_d38.py": "02c9586b0f86fbb58a0a2e5f712bc141a058f72919d8c8adf6628ded33d273c3",
    "stream_contract_d37.py": "4d54498bf78e21c2fefa256c296c881948c07b7c5f68a3ff4cbc84ba738ce51f",
    "cpu_checkpoint_policy_d38.py": "66b9eab15acf3e10ae167c7a75fa05c77d9d231f8b7a4c32425216a9da7908f2",
}


def read(path, rank, baseline, baseline_path):
    prefix = "home/nik/bc250work/evidence/distributed-d38-r1/"
    source = "home/nik/bc250work/tools/distributed-d38/"
    with tarfile.open(baseline_path) as old:
        original_manifest = json.load(
            old.extractfile(
                "home/nik/bc250work/evidence/distributed-d35-r1/cpu-reference/reference/manifest.json"
            )
        )
    with tarfile.open(path) as archive:

        def blob(name):
            return archive.extractfile(prefix + name).read()

        def record(name):
            return json.loads(blob(name))

        handoff = record("handoff.json")
        assert (
            handoff["status"] == "passed"
            and handoff["final_reference_blobs_reverified"]
        )
        assert record("CPU-reference-manifest.json") == original_manifest
        cpu = record("CPU-reference-worker.json")
        assert cpu == baseline["records"]["cpu-reference"]["worker"]
        assert (
            record("CPU-reference-final.json")
            == baseline["records"]["cpu-reference"]["final"]
        )
        assert (
            hashlib.sha256(blob("CPU-reference-manifest.json")).hexdigest()
            == cpu["manifest_sha256"]
        )
        assert original_manifest["sources"] == {
            n: SOURCE_SHA256[n] for n in ORACLE_FILES
        }
        entries = {e["name"]: e for e in original_manifest["tensors"]}
        assert len(entries) == 148
        bound = max(e["gradient"]["bytes"] for e in entries.values())
        assert not any(m.name.endswith(".raw") for m in archive.getmembers())
        uuids, cids = set(), set()
        workers = {}
        rows = []
        old_cids = {r["final"]["container_id"] for r in baseline["records"].values()}
        for label in LABELS:
            worker, final = record(label + "/rank.json"), record(label + "/final.json")
            assert (
                handoff["records"][label] == dict(worker=worker, final=final)
                and final["worker"] == worker
            )
            assert (
                worker["status"] == final["status"] == "passed"
                and worker["config"] == CONFIG
            )
            assert (
                worker["physical_rank"] == rank and worker["parameters"] == PARAMETERS
            )
            streaming = True
            enabled = label == "checkpoint-stream"
            assert (
                worker["variant"]
                == final["variant"]
                == ("stream" if streaming else "retained")
            )
            assert (
                worker["cuda_initialized"] is False
                and worker["CPU_checkpointing_enabled"] is enabled
            )
            assert worker["torch"] == cpu["torch"] and len(worker["checks"]) == 8
            assert (
                worker["CPU_artifact_manifest_sha256"]
                == final["CPU_artifact_manifest_sha256"]
                == cpu["manifest_sha256"]
            )
            assert (
                worker["checkpoint"]
                == final["checkpoint"]
                == ("nonreentrant" if enabled else "off")
            )
            if enabled:
                validate_options(worker["checkpoint_options"])
            else:
                assert worker["checkpoint_options"] is None
            assert worker["checkpoint_runtime_modules"] == CONFIG["blocks"] + 1
            assert worker["block_calls_after_forward"] == {
                str(i): 1 for i in range(CONFIG["blocks"])
            }
            assert worker["block_calls_after_backward"] == {
                str(i): (2 if enabled else 1) for i in range(CONFIG["blocks"])
            }
            assert worker["CPU_rng_before_sha256"] == worker["CPU_rng_after_sha256"]
            assert worker["checkpoint_source_sha256"] == CHECKPOINT_SOURCE_SHA256
            assert worker["CPU_FP64_loss"] == cpu["CPU_FP64_loss"]
            assert (
                worker["gradient_parameters"] == 148
                and worker["gradient_elements"] == PARAMETERS
            )
            assert worker["initial_sha256"] == {
                n: e["initial"]["sha256"] for n, e in entries.items()
            }
            assert worker["gradient_sha256"] == {
                n: e["gradient"]["sha256"] for n, e in entries.items()
            }
            assert worker["hook_counts"] == {n: int(streaming) for n in entries}
            assert worker["retained_gradient_elements_after_backward"] == (
                0 if streaming else PARAMETERS
            )
            assert worker["maximum_visible_gradient_elements_at_hook"] == (
                bound // 8 if streaming else 0
            )
            assert worker["maximum_live_scratch_bytes"] == (bound if streaming else 0)
            assert (
                worker["scratch_removed"]
                and worker["storage"]["maximum_scratch_bytes"] == bound
            )
            assert worker["storage"]["free_before_bytes"] >= bound + 5 * GIB
            assert final["storage"]["maximum_scratch_bytes"] == bound
            assert worker["sources"] == SOURCE_SHA256
            for filename, wanted in SOURCE_SHA256.items():
                assert (
                    hashlib.sha256(
                        archive.extractfile(source + filename).read()
                    ).hexdigest()
                    == wanted
                )
            assert (
                final["registry_sha256"]
                == baseline["records"]["cpu-reference"]["final"]["registry_sha256"]
            )
            assert (
                final["registry_unchanged"]
                and final["kernel_log_unchanged"]
                and final["agent_shutdown_clean"]
            )
            assert final["docker_containers"] == dict(before=[], after=[])
            assert blob(label + "/kernel-before.log") == blob(
                label + "/kernel-after.log"
            )
            state, ack = record(label + "/guard/state.json"), record(
                label + "/guard/ack.json"
            )
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            assert state["boot_id"] == handoff["boot_id"] == baseline["boot_id"]
            assert (
                worker["birth"]["uuid"] not in uuids
                and final["container_id"] not in cids | old_cids
            )
            uuids.add(worker["birth"]["uuid"])
            cids.add(final["container_id"])
            trace = worker["memory_samples"]
            assert trace and min(s["host_available_bytes"] for s in trace) > 3 * GIB
            assert all(
                worker["cgroup_events_before"][k]
                == worker["cgroup_events_after"][k]
                == 0
                for k in ["oom", "oom_kill"]
            )
            assert (
                max(s["host_swap_used_bytes"] for s in trace)
                == max(s["memory.swap.current"] for s in trace)
                == 0
            )
            snapshots = {s["stage"]: s for s in worker["snapshots"]}
            assert set(snapshots) == {
                "before-model",
                "after-model",
                "after-initial-reference-check",
                "after-forward",
                "after-backward",
                "after-reference-check",
            }
            rows.append(
                dict(
                    host=rank,
                    label=label,
                    whole_cgroup_peak_MiB=max(s["memory.peak"] for s in trace)
                    / 1024**2,
                    process_peak_RSS_MiB=max(s["VmHWM"] for s in trace) / 1024**2,
                    sampled_RSS_max_MiB=max(s["VmRSS"] for s in trace) / 1024**2,
                    host_min_available_GiB=min(s["host_available_bytes"] for s in trace)
                    / GIB,
                    after_backward_cgroup_MiB=snapshots["after-backward"][
                        "memory.current"
                    ]
                    / 1024**2,
                    after_backward_RSS_MiB=snapshots["after-backward"]["VmRSS"]
                    / 1024**2,
                    retained_gradient_elements=worker[
                        "retained_gradient_elements_after_backward"
                    ],
                    maximum_live_scratch_bytes=worker["maximum_live_scratch_bytes"],
                    host_swap_max_bytes=max(s["host_swap_used_bytes"] for s in trace),
                    cgroup_swap_max_bytes=max(s["memory.swap.current"] for s in trace),
                    worker_seconds=worker["worker_seconds"],
                    guarded_phase_seconds=final["guarded_phase_seconds"],
                )
            )
            workers[label] = worker
        for key in [
            "CPU_FP64_loss",
            "initial_sha256",
            "gradient_sha256",
            "checkpoint_source_sha256",
            "CPU_rng_before_sha256",
            "CPU_rng_after_sha256",
        ]:
            assert workers[LABELS[0]][key] == workers[LABELS[1]][key]
        return handoff, workers, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "host104",
        "host105",
        "baseline104",
        "baseline105",
        "baseline-results",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ["stream-baseline104", "stream-baseline105", "stream-baseline-results"]:
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
    stream_published = json.loads(args.stream_baseline_results.read_text())
    assert stream_published["status"] == "passed" and stream_published["groups"] == 24
    stream_baselines = []
    for rank, path in enumerate([args.stream_baseline104, args.stream_baseline105]):
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == stream_published["archives"][rank]["sha256"]
        )
        _, workers, _ = read_d37(
            path, rank, baselines[rank], [args.baseline104, args.baseline105][rank]
        )
        stream_baselines.append(workers["stream-gradient"])
    left, lw, lr = read(args.host104, 0, baselines[0], args.baseline104)
    right, rw, rr = read(args.host105, 1, baselines[1], args.baseline105)
    for label in LABELS:
        for key in [
            "CPU_FP64_loss",
            "initial_sha256",
            "gradient_sha256",
            "hook_counts",
        ]:
            assert lw[label][key] == rw[label][key]
    for current, prior in zip(
        [lw["stream-control"], rw["stream-control"]], stream_baselines
    ):
        for key in [
            "CPU_FP64_loss",
            "initial_sha256",
            "gradient_sha256",
            "hook_counts",
            "retained_gradient_elements_after_backward",
            "maximum_live_scratch_bytes",
        ]:
            assert current[key] == prior[key]
    result = dict(
        status="passed",
        groups=32,
        guards=4,
        local_tests=86,
        config=CONFIG,
        parameters=PARAMETERS,
        CPU_only=True,
        fresh_CPU_oracle_executed=True,
        CPU_reference_reused_from_D35=True,
        exact_ALL_gradients_initial_loss=True,
        post_accumulate_streaming_qualified=True,
        native_nonreentrant_CPU_checkpointing_qualified=True,
        actual_block_recomputation_qualified=True,
        native_checkpoint_source_sha256=lw["stream-control"][
            "checkpoint_source_sha256"
        ],
        stream_baseline_archives=stream_published["archives"],
        GPU_training_changed=False,
        larger_than_single_fit_qualified=False,
        larger_CPU_oracle_admitted=False,
        PLAN05_complete=False,
        rows=lr + rr,
        boot_ids=[left["boot_id"], right["boot_id"]],
        baseline_archives=published["archives"],
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
