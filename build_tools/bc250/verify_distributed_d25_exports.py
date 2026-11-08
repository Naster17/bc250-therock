#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D25 recorded numerical/reference, scaler/skip and fresh replay results."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from export_distributed_d25_on_host import PHASES

COUNTS = {
    "fp16-seed": 14,
    "fp16-resume": 7,
    "overflow-seed": 14,
    "overflow-resume": 9,
    "inactive-seed": 4,
    "fp32-control-seed": 4,
    "bf16-control-seed": 4,
    "late-refusal": 5,
    "scale-refusal": 3,
}


def read(path):
    with tarfile.open(path) as t:
        base = "home/nik/bc250work/evidence/distributed-d25-"
        h = json.load(t.extractfile(base + "r3/handoff.json"))
        assert h["status"] == "passed" and h["original_checkpoint_files_unchanged"]
        for series, label, accepted in PHASES:
            phase = base + series + "/" + label + "/"
            rec = h["records"][series + "/" + label]
            w, f = rec["worker"], rec["final"]
            assert w == json.load(
                t.extractfile(phase + "rank.json")
            ) and f == json.load(t.extractfile(phase + "final.json"))
            assert w["status"] == f["status"] == ("passed" if accepted else "failed")
            assert f["kernel_log_unchanged"] and f["registry_unchanged"]
            assert f["docker_containers"] == dict(before=[], after=[]) and not f.get(
                "docker_error"
            )
            assert (
                t.extractfile(phase + "kernel-before.log").read()
                == t.extractfile(phase + "kernel-after.log").read()
            )
            state = json.load(t.extractfile(phase + "guard/state.json"))
            ack = json.load(t.extractfile(phase + "guard/ack.json"))
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            if not accepted:
                assert "AssertionError" in w["error"]
                continue
            assert (
                len(w["checks"]) == COUNTS[label]
                and len(w["CPU_policy_contracts"]) == 8
            )
            assert f["worker"] == w and f["agent_shutdown_clean"]
            source = "home/nik/bc250work/tools/distributed-d25/"
            for filename, sha in [
                ("distributed_d25_trainer.py", w["source_sha256"]),
                ("bc250_trainer_fsdp.py", w["config"]["helper_sha256"]),
            ]:
                assert (
                    hashlib.sha256(t.extractfile(source + filename).read()).hexdigest()
                    == sha
                )
            fp32 = label == "fp32-control-seed"
            bf16 = label == "bf16-control-seed"
            assert w["local_shard_parameter_elements"] == (
                [11282, 11279] if fp32 else [11284, 11277]
            )
            for error in w.get("gradient_errors", []):
                assert error["CPU_AMP"] <= (0.02 if bf16 else 0.004)
                assert error["CPU_FP64"] <= (0.03 if bf16 else 0.006)
            if label.endswith("refusal"):
                assert w["raw_updates"] == 0
                continue
            types = w["observed_dtypes"]
            dtype = (
                "torch.float32"
                if fp32
                else "torch.bfloat16" if bf16 else "torch.float16"
            )
            assert types["linear_outputs"] == types["forward_weights"] == [dtype]
            assert types["reduction_dtype"] == "torch.float32"
            assert types["scaler"] == (None if fp32 or bf16 else "ShardedGradScaler")
            gathers = w["parameter_allgathers"]
            assert len(gathers) == (
                35
                if label.endswith("seed") and label.startswith(("fp16", "overflow"))
                else 15 if label.endswith("resume") else 10
            )
            assert all(
                g["dtype"] == dtype
                and g["input_bytes"] == g["elements"] * (4 if fp32 else 2)
                for g in gathers
            )
            windows = (
                4
                if label in ["fp16-seed", "overflow-seed"]
                else 2 if label.endswith("resume") else 1
            )
            assert types["reduction_calls"] == windows * 3
            assert types["reduction_input_bytes"] == windows * (
                90256 if fp32 else 90272
            )
            if not fp32 and not bf16:
                scaler = types["final_scaler"]
                assert scaler["scale"] == (
                    128
                    if label.startswith("overflow")
                    else 512 if label.startswith("fp16") else 128
                )
                assert scaler["_growth_tracker"] == (
                    1
                    if label.startswith("overflow") or label.startswith("inactive")
                    else 0
                )
        for case in ["fp16", "overflow"]:
            seed = h["records"]["r3/" + case + "-seed"]["worker"]
            resume = h["records"]["r3/" + case + "-resume"]["worker"]
            assert (
                seed["birth"]["uuid"] != resume["birth"]["uuid"]
                and seed["birth"]["start_ticks"] != resume["birth"]["start_ticks"]
            )
            assert (
                resume["restore_optimizer_actual_keys"]
                == resume["restore_optimizer_expected_keys"]
            )
            assert resume["transcript"] == [
                v for v in seed["transcript"] if v["step"] > 2
            ]
            assert resume["final_state_digest"] == seed["final_state_digest"]
            assert resume["clip_records"] == seed["clip_records"][2:]
            if case == "overflow":
                assert (
                    seed["raw_rank_nonfinite_flags"]
                    == resume["raw_rank_nonfinite_flags"]
                    == [False, True]
                )
            assert len(h["payload_files"][case + "-seed"]) == 6
        return h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    nodes = [read(a.directory / f"{host}-evidence-r3.tar.gz") for host in [104, 105]]
    assert nodes[0]["payload_files"] == nodes[1]["payload_files"]
    for series, label, accepted in PHASES:
        if not accepted:
            continue
        workers = [n["records"][series + "/" + label]["worker"] for n in nodes]
        assert [w["rank"] for w in workers] == [0, 1]
        assert (
            workers[0]["config"] == workers[1]["config"]
            and workers[0]["source_sha256"] == workers[1]["source_sha256"]
        )
    result = dict(
        status="passed",
        accepted_GPU_groups=2 * sum(COUNTS.values()),
        accepted_GPU_guards=18,
        retained_failed_guards=4,
        local_unit_tests=58,
        CPU_policy_contract_patterns=8,
        boot_ids=[n["boot_id"] for n in nodes],
        kernels=[n["kernel"] for n in nodes],
        records={str(host): n["records"] for host, n in zip([104, 105], nodes)},
        payload_files=nodes[0]["payload_files"],
        payload_inventories={
            str(host): n["payload_inventory"] for host, n in zip([104, 105], nodes)
        },
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
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["records", "payload_inventories"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
