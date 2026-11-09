#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail closed unless the selected worker uses immutable D32 qualification math."""
import hashlib
import json
from pathlib import Path

PROOF_SHA256 = {
    0: "b3c89af9d505b4f9bf78fbffd1dc5a2b487c9a29b93c8439c2948cbdfd5ae887",
    1: "5e21ea99d383cca872042a663b5ef1b1d607510d8aa9313ce11190e6bcd8afc7",
}
SOURCES = {
    "distributed_d32_capacity.py",
    "capacity_plan_d32.py",
    "distributed_d05_train.py",
}


# Same modeling identities already audited in D29/D31; avoid importing Trainer.
MODELING_PINS = {
    "modeling_utils.py": "2f45c2139ce633e6e3eae9290675dd5815643b37c27f2cc4dd83586fb648f7bb",
    "modeling_layers.py": "9e4726c07cfc895d95f9a75b0e7d23b8561575080811695e70fa879b0a18dae4",
    "models/gpt2/modeling_gpt2.py": "97c3e08dadffb3e54a6c75d935af995b2cc9ba723fc49c16eb21056d49ee4ebd",
}


def load_qualification(mode, physical_rank, directory=None):
    if mode not in {"single", "ddp", "fsdp"} or physical_rank not in PROOF_SHA256:
        raise ValueError("Unknown qualification mode/rank")
    directory = Path(__file__).parent if directory is None else Path(directory)
    path = directory / f"distributed_d33_qualification_{physical_rank}.json"
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != PROOF_SHA256[physical_rank]:
        raise ValueError("Qualification evidence changed; requalify before GPU work")
    proof = json.loads(blob)
    if (
        proof["schema"] != 1
        or proof["physical_rank"] != physical_rank
        or set(proof["source_sha256"]) != SOURCES
    ):
        raise ValueError("Qualification schema/source set changed")
    for name, wanted in proof["source_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != wanted:
            raise ValueError("Qualified training source changed: " + name)
    selected = proof["reference"][mode]
    if (
        not selected["reference"]["ALL_AdamW_parameters_moments_counters"]
        or selected["reference"]["gradient_elements"] != 45871104
    ):
        raise ValueError("Incomplete original CPU qualification")
    return proof, selected
