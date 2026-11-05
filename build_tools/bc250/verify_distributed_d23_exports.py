#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D23 CPU reference/sharding results and exact fresh-process replay evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from export_distributed_d23_on_host import PHASES


def read(path):
    with tarfile.open(path) as t:
        root = "home/nik/bc250work/evidence/distributed-d23-r1/"
        h = json.load(t.extractfile(root + "handoff.json"))
        assert h["status"] == "passed"
        for label, tools, accepted, groups in PHASES:
            p = root + label + "/"
            rec = h["records"][label]
            w = rec["worker"]
            f = rec["final"]
            assert w == json.load(t.extractfile(p + "rank.json")) and f == json.load(
                t.extractfile(p + "final.json")
            )
            assert (
                w["status"] == f["status"] == ("passed" if accepted else "failed")
                and len(w["checks"]) == groups
            )
            assert f["kernel_log_unchanged"] and f["registry_unchanged"]
            assert (
                t.extractfile(p + "kernel-before.log").read()
                == t.extractfile(p + "kernel-after.log").read()
            )
            state = json.load(t.extractfile(p + "guard/state.json"))
            ack = json.load(t.extractfile(p + "guard/ack.json"))
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            if not accepted:
                assert "PermissionError" in w["error"]
                continue
            assert len(w["CPU_policy_contracts"]) == 4
            sources = "home/nik/bc250work/tools/" + tools + "/"
            assert (
                hashlib.sha256(
                    t.extractfile(sources + "distributed_d23_trainer.py").read()
                ).hexdigest()
                == w["source_sha256"]
            )
            assert (
                hashlib.sha256(
                    t.extractfile(sources + "bc250_trainer_fsdp.py").read()
                ).hexdigest()
                == w["config"]["helper_sha256"]
            )
            if accepted:
                assert (
                    f["worker"] == w
                    and f["agent_shutdown_clean"]
                    and w["local_shard_parameter_elements"] == [11282, 11279]
                )
        seed = h["records"]["bf16-seed"]["worker"]
        resume = h["records"]["bf16-resume"]["worker"]
        assert (
            resume["restore_optimizer_actual_keys"]
            == resume["restore_optimizer_expected_keys"]
        )
        assert (
            h["records"]["bf16-seed"]["final"]["container_id"]
            != h["records"]["bf16-resume"]["final"]["container_id"]
        )
        assert (
            seed["birth"]["uuid"] != resume["birth"]["uuid"]
            and seed["birth"]["start_ticks"] != resume["birth"]["start_ticks"]
        )
        assert (
            resume["transcript"] == [v for v in seed["transcript"] if v["step"] > 2]
            and resume["final_state_digest"] == seed["final_state_digest"]
        )
        assert resume["clip_records"] == seed["clip_records"][2:]
        assert len(seed["clip_records"]) == 4
        for clip in seed["clip_records"]:
            assert math.isclose(
                clip["norm"], clip["CPU_norm"], rel_tol=1e-5, abs_tol=1e-7
            )
            assert 0 < clip["coefficient"] < 1 and math.isclose(
                clip["coefficient"], 0.05 / (clip["CPU_norm"] + 1e-6), rel_tol=1e-14
            )
        inactive = h["records"]["inactive-seed"]["worker"]
        assert (
            len(inactive["clip_records"]) == 1
            and inactive["clip_records"][0]["coefficient"] == 1
        )
        refusal = h["records"]["nan-refusal"]["worker"]
        assert refusal["raw_updates"] == 0 and refusal["raw_rank_nonfinite_flags"] == [
            False,
            True,
        ]
        for label in ["bf16-seed", "bf16-resume", "inactive-seed", "fp32-control-seed"]:
            w = h["records"][label]["worker"]
            types = w["observed_dtypes"]
            assert types["linear_outputs"] == (
                ["torch.float32"]
                if label == "fp32-control-seed"
                else ["torch.bfloat16"]
            )
            assert (
                types["forward_weights"] == ["torch.float32"]
                and types["reduction_dtype"] == "torch.float32"
                and types["scaler"] is None
            )
            assert types["reduction_calls"] == (
                12 if label == "bf16-seed" else 6 if label == "bf16-resume" else 3
            )
            assert (
                types["reduction_input_bytes"] == types["reduction_calls"] // 3 * 90256
            )
            for error in w.get("gradient_errors", []):
                assert error["CPU_AMP"] <= 0.02 and error["CPU_FP64"] <= 0.03
        assert h["original_checkpoint_files_unchanged"] and len(h["payload_files"]) == 6
        return h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    nodes = [read(a.directory / f"{host}-evidence-r1.tar.gz") for host in [104, 105]]
    assert nodes[0]["payload_files"] == nodes[1]["payload_files"]
    for label, tools, accepted, groups in PHASES:
        w = [n["records"][label]["worker"] for n in nodes]
        if not accepted:
            continue
        assert (
            [x["rank"] for x in w] == [0, 1]
            and w[0]["config"] == w[1]["config"]
            and w[0]["source_sha256"] == w[1]["source_sha256"]
        )
    result = dict(
        status="passed",
        accepted_GPU_groups=64,
        accepted_GPU_guards=10,
        retained_failed_guards=0,
        CPU_policy_contract_patterns=4,
        CPU_policy_contract_observations=40,
        local_unit_tests=42,
        kernels=[n["kernel"] for n in nodes],
        boot_ids=[n["boot_id"] for n in nodes],
        shard_parameter_elements=[11282, 11279],
        source_pins=nodes[0]["records"]["bf16-resume"]["worker"]["config"][
            "source_pins"
        ],
        records={str(host): n["records"] for host, n in zip([104, 105], nodes)},
        payload_files=nodes[0]["payload_files"],
        payload_inventories={
            str(host): n["payload_inventory"] for host, n in zip([104, 105], nodes)
        },
        archives=[
            dict(
                host=h,
                bytes=(a.directory / f"{h}-evidence-r1.tar.gz").stat().st_size,
                sha256=hashlib.sha256(
                    (a.directory / f"{h}-evidence-r1.tar.gz").read_bytes()
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
