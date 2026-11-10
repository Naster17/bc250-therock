#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay D49 source bindings, raw profiler gates and activation/rollback records."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

NEW = "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b"
OLD = "a6dc1c884c4f446e23a2afc8185b26ef9fd55b57cdd1d15324d46054b902b9c2"
HELPER = "d33f2504de50eac21baf6f43ff4eb30a05c602881cda9b12b3df13360a2c3588"
BASE = "/opt/bc250-stack/candidates/"
SDK = "lib/librocprofiler-sdk.so.1.4.2"


def verify(archive):
    with tarfile.open(archive) as stream:
        raw = {
            m.name: stream.extractfile(m).read()
            for m in stream.getmembers()
            if m.isfile()
        }

    def get(name):
        return json.loads(raw[name])

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    controller = get("evidence/sdk-d49-controller/result.json")
    assert controller["status"] == "passed" and len(controller["steps"]) == 7
    assert all(step["exit"] == 0 for step in controller["steps"])
    counts = dict(processes=0, negative_imports=0, numerical_checks=0, gpu_traces=0)
    containers = set()
    samples = []
    seconds = {}
    for phase, expected_count in [
        ("activation-r1", 12),
        ("rollback-r1", 2),
        ("final-r1", 4),
    ]:
        root = "evidence/sdk-d49-" + phase
        result = get(root + "/result.json")
        assert result["status"] == "passed" and len(result["phases"]) == expected_count
        state = get(root + "/guard/state.json")
        ack = get(root + "/guard/ack.json")
        assert state == result["guard"] and state["status"] == "acknowledged"
        assert all(ack[key] == state[key] for key in ("boot_id", "nonce", "phase"))
        seconds[phase] = result["seconds"]
        for name, digest in result["sources"].items():
            assert sha(raw["tools/" + name]) == digest, name
        for item in result["phases"]:
            folder = root + "/" + item["name"]
            assert item["command"][0] == "/usr/local/bin/bc250-torch"
            assert item["exit"] == 0 and item["kernel_unchanged"]
            assert (
                raw[folder + "/kernel-before.log"] == raw[folder + "/kernel-after.log"]
            )
            worker = get(folder + "/worker.json")
            assert worker == item["worker"] and worker["status"] == "passed"
            assert worker["sdk_sha256"] == (OLD if phase == "rollback-r1" else NEW)
            assert worker["sdk_path"].startswith("/workspace/build/dist/rocm/lib/")
            counts["processes"] += 1
            counts["numerical_checks"] += len(worker["checks"])
            assert item["containers"] and not (set(item["containers"]) & containers)
            containers.update(item["containers"])
            samples.extend(item["samples"])
            if worker["kind"] == "cpu":
                assert worker["device_count"] == 0
                counts["negative_imports"] += 1
            else:
                assert (
                    worker["device_count"] == 1
                    and worker["architecture"] == "gfx1013:xnack-"
                )
                for name in ("gpu-0", "gpu-1"):
                    events = get(folder + "/" + name + ".json")["traceEvents"]
                    kernels = [e for e in events if e.get("cat") == "kernel"]
                    copies = [e for e in events if e.get("cat") == "gpu_memcpy"]
                    runtime = [
                        e
                        for e in events
                        if e.get("cat") in ("cuda_runtime", "hip_runtime")
                    ]
                    assert kernels and len(copies) >= 2
                    assert all(
                        math.isfinite(e["dur"]) and e["dur"] > 0
                        for e in kernels + copies
                    )

                    def ids(values):
                        return {
                            e.get("args", {}).get("correlation") for e in values
                        } - {None}

                    assert ids(kernels) & ids(runtime) and ids(copies) & ids(runtime)
                    counts["gpu_traces"] += 1
                profile = (
                    "rccl-single" if item["name"].startswith("rccl-single") else "daily"
                )
                baseline = get(
                    "evidence/sdk-d49-rollback-r1/" + profile + "-default/worker.json"
                )
                assert worker["digests"] == baseline["digests"]
        if phase == "rollback-r1":
            assert result["legacy_refusal"]["exit"] == 2
    assert counts == dict(
        processes=18, negative_imports=8, numerical_checks=138, gpu_traces=20
    )
    for sample in samples:
        assert (
            sample["available_bytes"] >= 3 * 1024**3
            and sample["disk_free_bytes"] >= 5 * 1024**3
        )
    before = get("evidence/sdk-d49-install/activate-r1/before.json")
    rollback = get("evidence/sdk-d49-install/rollback-r1/result.json")
    final = get("evidence/sdk-d49-install/activate-r2/result.json")
    assert rollback["status"] == final["status"] == "passed"
    assert rollback["sdk_sha256"] == OLD and final["sdk_sha256"] == NEW
    assert rollback["distributed"] == get(
        "evidence/sdk-d49-install/activate-r1/registry.before.json"
    )
    assert rollback["helper_sha256"] == before["helpers"][0]["sha256"]
    assert final["helper_sha256"] == HELPER == sha(raw["tools/bc250_torch.py"])
    preservation = get("cohort/preservation.json")
    assert set(preservation["old"]) == set(preservation["new"])
    changed = [
        name
        for name in preservation["old"]
        if preservation["old"][name] != preservation["new"][name]
    ]
    assert changed == [SDK] == preservation["changed_paths"]
    stage = get("cohort/stage.json")
    for profile, digest in stage["profiles"].items():
        assert sha(raw["cohort/profiles/" + profile + "/runtime.json"]) == digest
        override = get("cohort/profiles/" + profile + "/native-overrides.json")
        assert (
            override == stage["native_override"] and override["library_sha256"] == NEW
        )
    assert final["distributed"]["runtime_sha256"] == stage["profiles"]["rccl-single"]
    assert final["distributed"]["release"] == BASE + "sdk-d49-r1/profiles/rccl-single"
    exported = get("evidence/sdk-d49-export/final.json")
    assert exported["sdk_sha256"] == NEW and exported["root_preserved"]
    return dict(
        status="passed",
        **counts,
        archive_sha256=sha(Path(archive).read_bytes()),
        helper_sha256=HELPER,
        sdk_sha256=NEW,
        old_sdk_sha256=OLD,
        min_host_available_bytes=min(s["available_bytes"] for s in samples),
        min_disk_free_bytes=min(s["disk_free_bytes"] for s in samples),
        seconds=seconds,
        final=exported
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.archive)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "processes", "numerical_checks", "gpu_traces")
            }
        )
    )


if __name__ == "__main__":
    main()
