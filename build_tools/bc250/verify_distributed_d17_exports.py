#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D17 archives, sources, native guards and cross-rank prediction/metric agreement."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from export_distributed_d17_on_host import LABELS


def read(path):
    with tarfile.open(path) as t:
        root = "home/nik/bc250work/evidence/distributed-d17-r1/"
        h = json.load(t.extractfile(root + "handoff.json"))
        assert h["status"] == "passed"
        assert set(h["records"]) == set(LABELS)
        for label, final in h["records"].items():
            assert final == json.load(t.extractfile(root + label + "/final.json"))
            w = json.load(t.extractfile(root + label + "/rank.json"))
            assert w == final["worker"]
            assert final["status"] == "passed" and w["status"] == "passed"
            assert all(
                final[k]
                for k in [
                    "agent_shutdown_clean",
                    "kernel_log_unchanged",
                    "registry_unchanged",
                ]
            )
            assert (
                t.extractfile(root + label + "/kernel-before.log").read()
                == t.extractfile(root + label + "/kernel-after.log").read()
            )
            state = json.load(t.extractfile(root + label + "/guard/state.json"))
            ack = json.load(t.extractfile(root + label + "/guard/ack.json"))
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            source = t.extractfile(
                "home/nik/bc250work/tools/distributed-d17/distributed_d17_evaluate.py"
            ).read()
            assert hashlib.sha256(source).hexdigest() == w["source_sha256"]
            for name, digest in w["config"]["helpers"].items():
                data = t.extractfile(
                    "home/nik/bc250work/tools/distributed-d17/" + name
                ).read()
                assert hashlib.sha256(data).hexdigest() == digest
            assert w["token_metrics"]["evaluated_examples"] == 19
            assert w["token_metrics"]["active_tokens"] == (
                120 if label.startswith("gpt2") else 126
            )
            assert w["native_loss_token_NLL_gap"] > 1e-6
            assert math.isclose(
                abs(w["native_eval_loss"] - w["token_metrics"]["token_nll"]),
                w["native_loss_token_NLL_gap"],
                abs_tol=1e-15,
            )
            assert len(w["checks"]) == 8 and len(w["CPU_contract_checks"]) == 5
        return h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    nodes = [read(a.directory / f"{host}-evidence-r1.tar.gz") for host in [104, 105]]
    for label in LABELS:
        left, right = [n["records"][label]["worker"] for n in nodes]
        assert [left["rank"], right["rank"]] == [0, 1]
        assert (
            left["config"] == right["config"]
            and left["prediction_digest"] == right["prediction_digest"]
            and left["token_metrics"] == right["token_metrics"]
        )
    result = dict(
        status="passed",
        accepted_GPU_check_groups=96,
        accepted_GPU_guards=12,
        CPU_contract_patterns=5,
        CPU_contract_observations=60,
        local_unit_tests=42,
        kernels=[n["kernel"] for n in nodes],
        boot_ids=[n["boot_id"] for n in nodes],
        cases={
            label: dict(
                token_metrics=nodes[0]["records"][label]["worker"]["token_metrics"],
                CPU_FP64_token_nll=nodes[0]["records"][label]["worker"][
                    "CPU_FP64_token_nll"
                ],
                native_eval_loss=nodes[0]["records"][label]["worker"][
                    "native_eval_loss"
                ],
                native_loss_token_NLL_gap=nodes[0]["records"][label]["worker"][
                    "native_loss_token_NLL_gap"
                ],
            )
            for label in LABELS
        },
        archives=[
            dict(
                host=host,
                bytes=(a.directory / f"{host}-evidence-r1.tar.gz").stat().st_size,
                sha256=hashlib.sha256(
                    (a.directory / f"{host}-evidence-r1.tar.gz").read_bytes()
                ).hexdigest(),
            )
            for host in [104, 105]
        ],
    )
    (a.directory / "verified-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
