#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pinned D42 per-rank proofs; no raw reference access or model execution."""
import json
from pathlib import Path
from artifact_d35 import digest
from capacity_plan_d42 import descriptor
from capacity_plan_d41 import CONFIG, CONTROL, parameters

DESCRIPTOR_SHA256 = "1d8fef22942169acbd759d3639a9730d4090dd68ebf211debb660aea3121e7e1"


def proof(shape, rank):
    if shape not in {"control", "large"} or rank not in {0, 1}:
        raise ValueError("Unknown shape or physical rank")
    path = Path(__file__).with_name("qualification_d43.json")
    if digest(path) != DESCRIPTOR_SHA256:
        raise ValueError("D42 qualification descriptor changed")
    record = json.loads(path.read_text())
    assert record["report"]["status"] == "passed"
    item = record["proofs"][f"{rank}-{shape}"]
    worker = item["worker"]
    config = CONTROL if shape == "control" else CONFIG
    assert worker["status"] == "passed" and worker["phase"] == "qualify"
    assert worker["mode"] == "fsdp" and worker["config"] == config
    assert worker["rank"] == worker["physical_rank"] == rank and worker["world"] == 2
    assert (
        worker["CPU_artifact_manifest_sha256"] == descriptor(shape)["manifest_sha256"]
    )
    ref = worker["reference"]
    assert (
        ref["gradient_elements"]
        == ref["CPU_AdamW_values_checked"]
        == parameters(config)
    )
    assert ref["ALL_AdamW_parameters_moments_counters"]
    assert worker["state"]["local_master_elements"] == parameters(config) // 2
    for name, pin in worker["sources"].items():
        if digest(Path(__file__).with_name(name)) != pin:
            raise ValueError("D42 source changed: " + name)
    return item


def sources(shape, rank):
    result = dict(proof(shape, rank)["worker"]["sources"])
    for name in [
        "qualification_d43.json",
        "qualification_d43.py",
        "distributed_d43_train.py",
    ]:
        result[name] = digest(Path(__file__).with_name(name))
    return result
