#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay D48 source/provider, visibility, exact math and native trace evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

from audit_sdk_d48_abi import contract

OLD = "a6dc1c884c4f446e23a2afc8185b26ef9fd55b57cdd1d15324d46054b902b9c2"
NEW = "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b"
SOURCE = "5aa899d0c9921077fb411f9d649e6735b2381a49c97a2fe4caa79db5bb03fcc0"


def verify(archive, old_sdk, new_sdk):
    old_data, new_data = old_sdk.read_bytes(), new_sdk.read_bytes()
    assert hashlib.sha256(old_data).hexdigest() == OLD
    assert hashlib.sha256(new_data).hexdigest() == NEW
    assert contract(old_data) == contract(new_data)
    with tarfile.open(archive) as bundle:

        def data(name):
            return bundle.extractfile(name).read()

        def record(name):
            return json.loads(data(name))

        build = record("evidence/sdk-d48-build-r2/build.json")
        assert build["status"] == "built-unqualified" and build["ci_preserved"]
        assert build["source_sha256"] == SOURCE and build["library_sha256"] == NEW
        assert (
            hashlib.sha256(data("evidence/sdk-d48-build-r2/agent.cpp")).hexdigest()
            == SOURCE
        )
        assert "-DROCPROFILER_CI=1" in build["compile_argv"]
        assert "--dependency-file=/workspace/d48/candidate-link.d" in build["link_argv"]
        result = record("evidence/sdk-d48-tests-r3/result.json")
        assert (
            result["status"] == "passed" and result["guard"]["status"] == "acknowledged"
        )
        state = record("evidence/sdk-d48-tests-r3/guard/state.json")
        ack = record("evidence/sdk-d48-tests-r3/guard/ack.json")
        assert state == result["guard"]
        assert all(ack[k] == state[k] for k in ("boot_id", "phase", "nonce"))
        for name, digest in result["sources"].items():
            assert (
                hashlib.sha256(data("tools/sdk-d48/" + name)).hexdigest() == digest
            ), name
        assert result["candidate_sha256"] == NEW and result["active_sdk_sha256"] == OLD
        assert len(result["phases"]) == 23
        controls = {}
        for phase in result["phases"]:
            if phase["name"] in ("daily-control", "rccl-single-control"):
                controls[phase["name"].removesuffix("-control")] = phase["worker"]
        assert set(controls) == {"daily", "rccl-single"}
        gpu_traces, numeric, negative, cids = 0, 0, [], []
        for phase in result["phases"]:
            folder = "evidence/sdk-d48-tests-r3/" + phase["name"] + "/"
            worker = record(folder + "worker.json")
            assert worker == phase["worker"] and worker["status"] == "passed"
            assert phase["exit"] == 0 and phase["kernel_unchanged"]
            assert data(folder + "kernel-before.log") == data(
                folder + "kernel-after.log"
            )
            assert len(phase["containers"]) == 1
            cids.extend(phase["containers"])
            assert all(
                s["available_bytes"] >= 3 * 1024**3
                and s["disk_free_bytes"] >= 5 * 1024**3
                for s in phase["samples"]
            )
            assert worker["sdk_sha256"] == (OLD if phase["control"] else NEW)
            assert worker["sdk_path"].startswith(
                "/workspace/build/dist/rocm/lib/"
                if phase["control"]
                else "/workspace/sdk-d48/lib/"
            )
            numeric += len(worker["checks"])
            if worker["kind"] == "cpu":
                assert worker["device_count"] == 0 and len(worker["captures"]) == 1
                assert phase["value"] == "-1"
                negative.append(phase["name"])
            else:
                assert (
                    worker["device_count"] == 1
                    and worker["architecture"] == "gfx1013:xnack-"
                )
                assert worker["uuid"] == "GPU-XX"
                profile = (
                    "rccl-single"
                    if phase["name"].startswith("rccl-single-")
                    else "daily"
                )
                assert worker["digests"] == controls[profile]["digests"]
                assert len(worker["captures"]) == 3
            for capture in worker["captures"]:
                trace = record(folder + capture["name"] + ".json")["traceEvents"]
                kernels = [e for e in trace if e.get("cat") == "kernel"]
                copies = [e for e in trace if e.get("cat") == "gpu_memcpy"]
                runtime = [
                    e for e in trace if e.get("cat") in ("cuda_runtime", "hip_runtime")
                ]
                assert (
                    len(kernels) == capture["kernels"]
                    and len(copies) == capture["copies"]
                )
                if (
                    capture["name"].startswith("gpu")
                    and not worker["expected_missing_trace"]
                ):
                    assert kernels and len(copies) >= 2
                    assert all(
                        math.isfinite(e["dur"]) and e["dur"] > 0
                        for e in kernels + copies
                    )
                    ids = lambda events: {
                        e.get("args", {}).get("correlation") for e in events
                    } - {None}
                    assert ids(kernels) & ids(runtime) and ids(copies) & ids(runtime)
                    gpu_traces += 1
                else:
                    assert not kernels and not copies
        assert (
            len(set(cids)) == 23
            and len(negative) == 6
            and numeric == 227
            and gpu_traces == 32
        )
        expected_negatives = {
            p + "-" + alias + "-negative"
            for p in ("daily", "rccl-single")
            for alias in ("hip", "cuda", "rocr")
        }
        assert set(negative) == expected_negatives
        assert result["phases"][0]["worker"]["expected_missing_trace"]
        assert "GPU-XX" in result["uuid_selection"]
        final = record("evidence/sdk-d48-tests-r3/final-preflight.json")
        assert not final["docker_containers"] and not final["podman_containers"]
        return dict(
            status="passed",
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            candidate_sha256=NEW,
            old_sha256=OLD,
            source_sha256=SOURCE,
            processes=23,
            negative_imports=6,
            numeric_checks=227,
            gpu_traces=32,
            dynamic_symbols=len(contract(new_data)["symbols"]),
            uuid_selection_qualified=False,
            active_prefix_changed=False,
            historical_python_cpu_fault_fixed=False,
            clocksource_fixed=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--old-sdk", type=Path, required=True)
    parser.add_argument("--new-sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(verify(args.archive, args.old_sdk, args.new_sdk), indent=2) + "\n"
    )
