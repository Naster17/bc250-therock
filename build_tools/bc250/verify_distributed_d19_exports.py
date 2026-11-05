#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D19 archives, sources, native guards and cross-rank prediction/metric agreement."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from export_distributed_d19_on_host import LABELS


def read(path):
    with tarfile.open(path) as t:
        root = "home/nik/bc250work/evidence/distributed-d19-r3/"
        h = json.load(t.extractfile(root + "handoff.json"))
        assert h["status"] == "passed"
        assert set(h["records"]) == set(LABELS)
        for label, final in h["records"].items():
            phase_root = root
            selected_tools = "home/nik/bc250work/tools/distributed-d19-r3/"
            assert final == json.load(t.extractfile(phase_root + label + "/final.json"))
            w = json.load(t.extractfile(phase_root + label + "/rank.json"))
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
                t.extractfile(phase_root + label + "/kernel-before.log").read()
                == t.extractfile(phase_root + label + "/kernel-after.log").read()
            )
            state = json.load(t.extractfile(phase_root + label + "/guard/state.json"))
            ack = json.load(t.extractfile(phase_root + label + "/guard/ack.json"))
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            source = t.extractfile(
                selected_tools + "distributed_d19_evaluate.py"
            ).read()
            assert hashlib.sha256(source).hexdigest() == w["source_sha256"]
            for name, digest in w["config"]["helpers"].items():
                data = t.extractfile(selected_tools + name).read()
                assert hashlib.sha256(data).hexdigest() == digest
            assert w["token_metrics"]["evaluated_examples"] == 19
            if "wide" not in label:
                assert w["token_metrics"]["active_tokens"] == (
                    120 if label.startswith("gpt2") else 126
                )
            assert len(w["GPU_contract_checks"]) == 2
            profiles = w["CPU_dispatch_profiles"]
            assert profiles["compact"] == {
                "aten::_local_scalar_dense": 1,
                "aten::nonzero": 3,
            }
            assert (
                profiles["d18"]["aten::_local_scalar_dense"] > 1
                and profiles["d18"]["aten::nonzero"] > 0
            )
            assert w["native_loss_token_NLL_gap"] > 1e-6
            assert math.isclose(
                abs(w["native_eval_loss"] - w["token_metrics"]["token_nll"]),
                w["native_loss_token_NLL_gap"],
                abs_tol=1e-15,
            )
            assert len(w["checks"]) == 11 and len(w["CPU_contract_checks"]) == 5
            samples = w["ABCCBA_samples"]
            assert [v["mode"] for v in samples] == [
                "full",
                "d18",
                "compact",
                "compact",
                "d18",
                "full",
            ]
            for sample in samples:
                assert sample["elapsed_s"] > 0 and sample["peak_allocated_bytes"] > 0
                assert sample["label_array_bytes"] == (
                    9728 if "wide" in label else 1824
                )
                assert sample["prediction_array_bytes"] == (
                    (159383552 if "wide" in label else 88464)
                    if sample["mode"] == "full"
                    else 456
                )
                assert sample["native_loss"] == w["native_eval_loss"]
                for key, value in w["token_metrics"].items():
                    assert math.isclose(
                        sample["metrics"][key], value, rel_tol=1e-11, abs_tol=1e-11
                    )
            assert (
                samples[0]["prediction_gather_input_bytes"]
                == samples[5]["prediction_gather_input_bytes"]
            )
            assert (
                samples[1]["prediction_gather_input_bytes"]
                == samples[2]["prediction_gather_input_bytes"]
                == samples[3]["prediction_gather_input_bytes"]
                == samples[4]["prediction_gather_input_bytes"]
                == 240
            )
            assert samples[0]["prediction_gather_input_bytes"] > 240
        failure_root = "home/nik/bc250work/evidence/distributed-d19-r1/"
        failed = h["failed_bf16_wide"]
        assert (
            failed["status"] == "failed"
            and failed["kernel_log_unchanged"]
            and failed["registry_unchanged"]
        )
        assert failed == json.load(
            t.extractfile(failure_root + "bertwide-bf16-eval/final.json")
        )
        failed_worker = json.load(
            t.extractfile(failure_root + "bertwide-bf16-eval/rank.json")
        )
        assert (
            failed_worker["status"] == "failed"
            and "Mismatched elements: 2 / 30408704" in failed_worker["error"]
        )
        state = json.load(
            t.extractfile(failure_root + "bertwide-bf16-eval/guard/state.json")
        )
        ack = json.load(
            t.extractfile(failure_root + "bertwide-bf16-eval/guard/ack.json")
        )
        assert state["status"] == "acknowledged" and all(
            state[k] == ack[k] for k in ["nonce", "boot_id", "phase"]
        )
        return h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    nodes = [read(a.directory / f"{host}-evidence-r3.tar.gz") for host in [104, 105]]
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
        accepted_GPU_check_groups=176,
        accepted_GPU_guards=16,
        CPU_contract_patterns=5,
        CPU_contract_observations=80,
        local_unit_tests=42,
        GPU_contract_patterns=2,
        GPU_contract_observations=32,
        retained_failed_BF16_wide_guards=2,
        failed_BF16_wide="Bert BF16 CPU FP64 full-logit strict gate:2/30408704 out of tolerance, maximum absolute difference0.003654425659018067; no tolerance change or qualification.",
        kernels=[n["kernel"] for n in nodes],
        boot_ids=[n["boot_id"] for n in nodes],
        cases={
            label: dict(
                token_metrics=nodes[0]["records"][label]["worker"]["token_metrics"],
                CPU_FP64_token_nll=nodes[0]["records"][label]["worker"][
                    "CPU_FP64_token_nll"
                ],
                observations={
                    str(host): nodes[i]["records"][label]["worker"]["ABCCBA_samples"]
                    for i, host in enumerate([104, 105])
                },
                CPU_dispatch_profiles=nodes[0]["records"][label]["worker"][
                    "CPU_dispatch_profiles"
                ],
                config=nodes[0]["records"][label]["worker"]["config"],
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
                bytes=(a.directory / f"{host}-evidence-r3.tar.gz").stat().st_size,
                sha256=hashlib.sha256(
                    (a.directory / f"{host}-evidence-r3.tar.gz").read_bytes()
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
