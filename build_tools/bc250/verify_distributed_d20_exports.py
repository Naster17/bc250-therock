#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify preserved D20 native BF16 diagnosis and explicit FP32-head qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from export_distributed_d20_on_host import PHASES


def read(path):
    with tarfile.open(path) as t:
        h = json.load(
            t.extractfile("home/nik/bc250work/evidence/distributed-d20-handoff-r3.json")
        )
        assert h["status"] == "passed" and set(h["records"]) == {"r1", "r2", "r3"}
        for series, label, tools, groups in PHASES:
            root = f"home/nik/bc250work/evidence/distributed-d20-{series}/{label}/"
            final = json.load(t.extractfile(root + "final.json"))
            w = json.load(t.extractfile(root + "rank.json"))
            assert final == h["records"][series] and final["worker"] == w
            assert (
                final["status"] == w["status"] == "passed"
                and len(w["checks"]) == groups
            )
            assert all(
                final[k]
                for k in [
                    "agent_shutdown_clean",
                    "kernel_log_unchanged",
                    "registry_unchanged",
                ]
            )
            assert (
                t.extractfile(root + "kernel-before.log").read()
                == t.extractfile(root + "kernel-after.log").read()
            )
            state = json.load(t.extractfile(root + "guard/state.json"))
            ack = json.load(t.extractfile(root + "guard/ack.json"))
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            source = f"home/nik/bc250work/tools/{tools}/"
            assert (
                hashlib.sha256(
                    t.extractfile(
                        source
                        + (
                            "distributed_d20_scope.py"
                            if series == "r3"
                            else "distributed_d20_precision.py"
                        )
                    ).read()
                ).hexdigest()
                == w["source_sha256"]
            )
            for name, sha in w["helpers"].items():
                assert (
                    hashlib.sha256(t.extractfile(source + name).read()).hexdigest()
                    == sha
                )
            if series == "r3":
                assert (
                    abs(w["metrics"]["native_token_nll"] - 10.432745588959701) < 1e-11
                )
                assert abs(w["metrics"]["head_token_nll"] - 10.43274511247336) < 1e-11
                continue
            assert (
                len(w["digest_contracts"]) == 3
                and len(w["prediction_digest"]) == 64
                and w["token_metrics"]["active_tokens"] == 838
            )
            for key in [
                "GPU_BF16_vs_CPU_FP64",
                "GPU_BF16_vs_matched_CPU_BF16",
                "CPU_BF16_vs_CPU_FP64",
            ]:
                assert w[key]["compared_values"] == 30408704 and [
                    w[key]["atol"],
                    w[key]["rtol"],
                ] == [0.003, 0.003]
            assert w["GPU_BF16_vs_matched_CPU_BF16"]["violations"] == 0
            if series == "r1":
                assert (
                    w["GPU_BF16_vs_CPU_FP64"]["violations"]
                    == w["CPU_BF16_vs_CPU_FP64"]["violations"]
                    == 2
                )
                assert (
                    w["GPU_BF16_vs_CPU_FP64"]["outliers"]
                    == w["CPU_BF16_vs_CPU_FP64"]["outliers"]
                )
                assert all(
                    v["actual"] == v["CPU_BF16"] for v in w["CPU_BF16_at_GPU_outliers"]
                )
            else:
                assert (
                    w["GPU_BF16_vs_CPU_FP64"]["violations"]
                    == w["CPU_BF16_vs_CPU_FP64"]["violations"]
                    == 0
                )
                assert len(w["head_contracts"]) == 4 and w["policy"] == "fp32-head"
                assert (
                    w["matched_taps"]["cls.predictions.decoder"]["GPU_dtype"]
                    == "torch.float32"
                )
        return h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    nodes = [read(a.directory / f"{host}-evidence-r3.tar.gz") for host in [104, 105]]
    for series in ["r1", "r2", "r3"]:
        left, right = [n["records"][series]["worker"] for n in nodes]
        assert [left["rank"], right["rank"]] == [0, 1]
        if series == "r3":
            assert (
                left["metrics"] == right["metrics"]
                and left["source_sha256"] == right["source_sha256"]
                and left["helpers"] == right["helpers"]
            )
            continue
        for k in [
            "source_sha256",
            "helpers",
            "prediction_digest",
            "token_metrics",
            "GPU_BF16_vs_CPU_FP64",
            "GPU_BF16_vs_matched_CPU_BF16",
            "CPU_BF16_vs_CPU_FP64",
        ]:
            assert left[k] == right[k], (series, k)
    result = dict(
        status="passed",
        GPU_check_groups=36,
        GPU_guards=6,
        CPU_digest_patterns=3,
        CPU_digest_observations=12,
        CPU_head_patterns=4,
        CPU_head_observations=8,
        local_unit_tests=42,
        kernels=[n["kernel"] for n in nodes],
        boot_ids=[n["boot_id"] for n in nodes],
        cases={s: nodes[0]["records"][s]["worker"] for s in ["r1", "r2", "r3"]},
        archives=[
            dict(
                host=h,
                bytes=(a.directory / f"{h}-evidence-r3.tar.gz").stat().st_size,
                sha256=hashlib.sha256(
                    (a.directory / f"{h}-evidence-r3.tar.gz").read_bytes()
                ).hexdigest(),
            )
            for h in [104, 105]
        ],
    )
    (a.directory / "verified-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
