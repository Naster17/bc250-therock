#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify D28 bounded frozen-policy correctness, refusal and replay archives."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from export_distributed_d28_on_host import PHASES

COUNTS = {
    "frozen-seed": (21, 20),
    "frozen-resume": (10, 10),
    "frozen-missing-seed": (4, 3),
    "frozen-grad-seed": (4, 3),
    "frozen-late-seed": (7, 6),
    "frozen-peft-seed": (7, 6),
    "retained-control-seed": (6, 5),
    "frozen-policy-resume": (2, 2),
}


def read(path):
    with tarfile.open(path) as t:
        prefix = "home/nik/bc250work/evidence/distributed-d28-"
        source = "home/nik/bc250work/tools/distributed-d28/"
        h = json.load(t.extractfile(prefix + "r3/handoff.json"))
        assert (
            h["status"] == "passed"
            and h["original_checkpoint_files_unchanged"]
            and h["rank1_only_private_policy_change_verified"]
        )
        groups = 0
        observations = 0
        for series, label, accepted in PHASES:
            rec = h["records"][series + "/" + label]
            w, f = rec["worker"], rec["final"]
            phase = prefix + series + "/" + label + "/"
            assert w == json.load(
                t.extractfile(phase + "rank.json")
            ) and f == json.load(t.extractfile(phase + "final.json"))
            assert w["status"] == f["status"] == ("passed" if accepted else "failed")
            assert (
                f["kernel_log_unchanged"]
                and f["registry_unchanged"]
                and f["docker_containers"] == dict(before=[], after=[])
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
            assert state["boot_id"] == h["boot_id"]
            if not accepted:
                assert "lora_A.default.weight" in w["error"]
                continue
            assert f["worker"] == w and f["agent_shutdown_clean"]
            if series != "r3":
                continue
            rank = w["rank"]
            assert len(w["checks"]) == COUNTS[label][rank]
            groups += len(w["checks"])
            observations += len(w["CPU_policy_contracts"]) + len(
                w["CPU_causal_contracts"]
            )
            for filename, sha in [
                ("distributed_d25_trainer.py", w["source_sha256"]),
                ("bc250_trainer_fsdp.py", w["config"]["helper_sha256"]),
            ]:
                assert (
                    hashlib.sha256(t.extractfile(source + filename).read()).hexdigest()
                    == sha
                )
            if label == "retained-control-seed":
                assert (
                    w["config"]["parameters"] == 24352
                    and w["config"]["trainable_parameters"] == 3072
                )
                assert w["local_shard_parameter_elements"] == [12176, 12176]
                assert w["local_trainable_elements"] == [1504, 1568]
                assert len(w["CPU_causal_contracts"]) == 14
                continue
            assert (
                len(w["CPU_policy_contracts"]) == 8
                and len(w["CPU_causal_contracts"]) == 18
            )
            for filename, sha in [
                ("distributed_d28_trainer.py", w["fixture_source_sha256"]),
                ("bc250_fsdp_frozen.py", w["frozen_helper_sha256"]),
                ("bc250_fsdp_peft.py", w["peft_helper_sha256"]),
            ]:
                assert (
                    hashlib.sha256(t.extractfile(source + filename).read()).hexdigest()
                    == sha
                )
            assert (
                w["config"]["parameters"] == 24352
                and w["config"]["trainable_parameters"] == 3072
            )
            assert w["frozen_policy_sha256"] == h["checkpoint_policy"]["policy_sha256"]
            if label == "frozen-policy-resume":
                assert (
                    w["raw_updates"] == 0 and "Frozen checkpoint policy" in w["refusal"]
                )
                continue
            assert w["local_shard_parameter_elements"] == [12176, 12176]
            assert w["local_trainable_elements"] == [1536, 1536]
            if label in [
                "frozen-missing-seed",
                "frozen-grad-seed",
                "frozen-late-seed",
                "frozen-peft-seed",
            ]:
                assert w["raw_updates"] == 0
                continue
            seed = label == "frozen-seed"
            windows = 4 if seed else 2
            assert w["optimizer_shard_elements"] == 1536
            assert w["tied_parameter_checks"] == (18 if seed else 9)
            assert w["frozen_gate_calls"] == windows
            assert [x["shifted"] for x in w["target_counts"]] == (
                [49, 57, 57, 21] if seed else [57, 21]
            )
            obs = w["observed_dtypes"]
            assert obs["scaler"] is None and obs["final_scaler"] == {}
            assert obs["linear_outputs"] == obs["forward_weights"] == ["torch.float32"]
            assert (
                obs["reduction_calls"] == windows * 16
                and obs["reduction_input_bytes"] == windows * 12288
            )
            gathers = w["parameter_allgathers"]
            assert len(gathers) == (259 if seed else 111)
            assert all(
                g["dtype"] == "torch.float32" and g["input_bytes"] == g["elements"] * 4
                for g in gathers
            )
            assert len(w["adapter_exports"]) == (2 if seed else 1)
            assert all(
                v["parameters_exact"]
                and v["logits_exact"]
                and v["rng_unchanged"]
                and v["adapter_bytes"] == 14216
                for v in w["adapter_exports"]
            )
            if not seed:
                old = h["records"]["r3/frozen-seed"]["worker"]
                assert (
                    w["birth"]["uuid"] != old["birth"]["uuid"]
                    and w["birth"]["start_ticks"] != old["birth"]["start_ticks"]
                )
                assert (
                    f["container_id"]
                    != h["records"]["r3/frozen-seed"]["final"]["container_id"]
                )
                assert w["transcript"] == [
                    v for v in old["transcript"] if v["step"] > 2
                ]
                assert w["final_state_digest"] == old["final_state_digest"]
                assert (
                    w["restore_optimizer_actual_keys"]
                    == w["restore_optimizer_expected_keys"]
                    and w["restore_optimizer_actual_keys"]
                )
        assert (
            len(h["payload_files"]) == 6
            and sum(x["bytes"] for x in h["payload_files"].values()) == 480247
        )
        return h, groups, observations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host104", type=Path, required=True)
    p.add_argument("--host105", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    left, lg, lo = read(a.host104)
    right, rg, ro = read(a.host105)
    assert left["checkpoint_policy"] == right["checkpoint_policy"]
    assert (
        left["records"]["r3/frozen-seed"]["worker"]["frozen_initial_digest"]
        == right["records"]["r3/frozen-seed"]["worker"]["frozen_initial_digest"]
    )
    for label in ["frozen-seed", "frozen-resume"]:
        lw = left["records"]["r3/" + label]["worker"]
        rw = right["records"]["r3/" + label]["worker"]
        for l, r in zip(lw["target_counts"], rw["target_counts"]):
            assert (
                sum(l["local_active"]) + sum(r["local_active"])
                == l["shifted"]
                == r["shifted"]
            )
    result = dict(
        status="passed",
        groups=lg + rg,
        final_guards=16,
        retained_failed_probe_guards=2,
        intermediate_passed_guards=2,
        all_guards_acknowledged=20,
        CPU_contract_patterns=26,
        CPU_contract_observations=lo + ro,
        parameters=24352,
        frozen_parameters=21280,
        trainable_parameters=3072,
        parameter_shards=[12176, 12176],
        trainable_and_AdamW_shards=[1536, 1536],
        frozen_parameters_exactly_preserved=True,
        fresh_all_state_replay_exact=True,
        refusals=[
            "rank1 missing trainable gradient",
            "rank1 injected frozen gradient",
            "rank1 post-clip NaN",
            "rank1-only private checkpoint-recipe corruption",
            "rank1 post-clip LoRA scaling mutation",
        ],
        checkpoint_payload_bytes=480247,
        checkpoint_policy_sha256=left["checkpoint_policy"]["policy_sha256"],
        boot_ids=[left["boot_id"], right["boot_id"]],
        native_runtime_changed=False,
        PEFT_version="0.18.1",
        native_full_model_checkpoint=True,
        public_adapter_roundtrip_exact=True,
        source_pins=left["checkpoint_policy"]["peft_source_pins"],
        archives=[
            dict(
                path=str(path),
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in [a.host104, a.host105]
        ],
        next="PLAN-04 / D29 planned: audit and qualify explicit non-reentrant FP32 PEFT activation checkpointing; retain frozen AMP guards and compare live allocator costs within safe headroom",
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
