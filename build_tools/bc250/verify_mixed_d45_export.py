#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay D45 source, reference coverage, reservation, resource and log checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


def verify(path):
    with tarfile.open(path) as archive:

        def data(name):
            return archive.extractfile(name).read()

        def record(name):
            return json.loads(data(name))

        root = "evidence/mixed-d45-r1/"
        result = record(root + "result.json")
        assert (
            result["status"] == "passed" and result["guard"]["status"] == "acknowledged"
        )
        guard = record(root + "guard/state.json")
        ack = record(root + "guard/ack.json")
        assert result["guard"] == guard
        assert all(ack[k] == guard[k] for k in ["boot_id", "phase", "nonce"])
        assert record(root + "export.json")["boot_id"] == result["boot_id"]
        for name, digest in result["sources"].items():
            assert (
                hashlib.sha256(data("tools/mixed-d45/" + name)).hexdigest() == digest
            ), name
        assert len(result["phases"]) == 6
        expected_names = [
            f"{op}.{field}"
            for op, fields in {
                "matmul": ["output", "loss", "gradient0", "gradient1"],
                "conv": ["output", "loss", "gradient0", "gradient1", "gradient2"],
                "sdpa": ["output", "loss", "gradient0", "gradient1", "gradient2"],
                "fft": ["output", "loss", "gradient0"],
                "adamw1": [
                    "loss",
                    "gradient",
                    "parameter",
                    "first_moment",
                    "second_moment",
                ],
                "adamw2": [
                    "loss",
                    "gradient",
                    "parameter",
                    "first_moment",
                    "second_moment",
                ],
            }.items()
            for field in fields
        ]
        rows = []
        containers = []
        for index, phase in enumerate(result["phases"]):
            profile = "daily" if index % 2 == 0 else "rccl-single"
            assert phase["name"] == f"{index}-{profile}" and phase["profile"] == profile
            assert phase["exit"] == 0 and phase["kernel_unchanged"]
            folder = root + phase["name"] + "/"
            assert data(folder + "kernel-before.log") == data(
                folder + "kernel-after.log"
            )
            worker = record(folder + "worker.json")
            assert worker == phase["worker"] and worker["status"] == "passed"
            assert worker["seed"] == 4500 + index and worker["optimizer_steps"] == 2
            assert worker["architecture"] == "gfx1013:xnack-"
            assert (worker["atol"], worker["rtol"]) == (5e-5, 5e-4)
            assert [c["name"] for c in worker["checks"]] == expected_names
            assert all(
                c["elements"] > 0 and math.isfinite(c["max_abs"]) and c["max_abs"] >= 0
                for c in worker["checks"]
            )
            events = [
                json.loads(line)
                for line in data(folder + "worker.log").decode().splitlines()
                if line.startswith("{")
            ]
            assert events[:-1] == [
                dict(stage=op, event=event)
                for op in ("matmul", "conv", "sdpa", "fft", "adamw")
                for event in ("begin", "passed")
            ]
            assert events[-1] == dict(status="passed", checks=27)
            assert len(phase["containers"]) == 1 and phase.get("cgroup")
            containers.extend(phase["containers"])
            assert phase["samples"] and phase["seconds"] < 60
            assert all(
                s["available_bytes"] >= 3 * 1024**3
                and s["disk_free_bytes"] >= 5 * 1024**3
                and s.get("memory.swap.current", 0) == s.get("oom_kill", 0) == 0
                for s in phase["samples"]
            )
            sampled = [s for s in phase["samples"] if "memory.peak" in s]
            assert sampled
            assert (
                not phase["postflight"]["docker_containers"]
                and not phase["postflight"]["podman_containers"]
            )
            rows.append(
                dict(
                    profile=profile,
                    seed=worker["seed"],
                    checks=27,
                    seconds=phase["seconds"],
                    gpu_allocated_peak=worker["gpu_allocated_peak"],
                    gpu_reserved_peak=worker["gpu_reserved_peak"],
                    sampled_cgroup_peak=max(s["memory.peak"] for s in sampled),
                    min_host_available=min(
                        s["available_bytes"] for s in phase["samples"]
                    ),
                )
            )
        assert len(set(containers)) == 6
        audit = record(root + "sdk-triage.json")
        matched = [
            b
            for b in audit["build_artifacts"]
            if b["sha256"] == audit["runtime_sdk_sha256"]
        ]
        return dict(
            status="passed",
            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rows=rows,
            reference_checks=162,
            fresh_processes=6,
            optimizer_updates=12,
            sdk_ci_build_matches_runtime=bool(matched),
            sdk_triage=audit,
            historical_python_fault_fixed=False,
            kfd_root_cause_fixed=False,
            clocksource_fixed=False,
            native_binaries_changed=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.archive), indent=2) + "\n")
