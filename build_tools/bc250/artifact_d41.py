#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D41 source identity and complete source-bound raw artifact verification."""
from pathlib import Path
from artifact_d35 import source_hashes as original_sources, digest, load_manifest
from capacity_plan_d41 import CONFIG, PARAMETERS

FILES = [
    "cpu_checkpoint_policy_d38.py",
    "stream_contract_d37.py",
    "capacity_plan_d41.py",
    "fixture_d41.py",
    "artifact_d41.py",
    "cpu_oracle_d41.py",
]


def sources():
    result = original_sources()
    result.update({n: digest(Path(__file__).with_name(n)) for n in FILES})
    return result


def validate(path, pin):
    result = load_manifest(path, pin, CONFIG, PARAMETERS, check_sources=False)
    if result["sources"] != sources() or len(result["tensors"]) != 244:
        raise ValueError("Unqualified D41 oracle source/coverage")
    return result
