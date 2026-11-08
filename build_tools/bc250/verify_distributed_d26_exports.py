#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check D26 archive identities, reference bounds, tied weights and exact replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from export_distributed_d26_on_host import PHASES


def read(path):
    with tarfile.open(path) as t:
        base = "home/nik/bc250work/evidence/distributed-d26-r1/"
        source = "home/nik/bc250work/tools/distributed-d26/"
        h = json.load(t.extractfile(base + "handoff.json"))
        assert h["status"] == "passed" and h["original_checkpoint_files_unchanged"]
        groups = 0
        for label in PHASES:
            rec = h["records"][label]
            w, f = rec["worker"], rec["final"]
            phase = base + label + "/"
            assert w == json.load(t.extractfile(phase + "rank.json"))
            assert f == json.load(t.extractfile(phase + "final.json"))
            assert w["status"] == f["status"] == "passed" and f["worker"] == w
            assert (
                f["kernel_log_unchanged"]
                and f["registry_unchanged"]
                and f["agent_shutdown_clean"]
            )
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
            assert state["boot_id"] == h["boot_id"]
            for filename, sha in [
                ("distributed_d25_trainer.py", w["source_sha256"]),
                ("bc250_trainer_fsdp.py", w["config"]["helper_sha256"]),
            ]:
                assert (
                    hashlib.sha256(t.extractfile(source + filename).read()).hexdigest()
                    == sha
                )
            groups += len(w["checks"])
            assert len(w["CPU_policy_contracts"]) == 8
            if label == "bert-control-seed":
                assert len(w["checks"]) == 4 and w[
                    "local_shard_parameter_elements"
                ] == [11282, 11279]
                continue
            entries = ["distributed_d26_trainer.py", "historical-source/entry-fp32.py"]
            assert any(
                hashlib.sha256(t.extractfile(source + n).read()).hexdigest()
                == w["fixture_source_sha256"]
                for n in entries
            )
            assert w["local_shard_parameter_elements"] == [10640, 10640]
            assert (
                w["optimizer_shard_elements"] == 10640
                and w["config"]["parameters"] == 21280
            )
            seed = label.endswith("seed")
            case = label.split("-")[0]
            expected_alias_checks = (
                (15 if case == "overflow" else 18)
                if seed
                else (7 if case == "overflow" else 9)
            )
            assert w["tied_parameter_checks"] == expected_alias_checks
            assert len(w["CPU_causal_contracts"]) == 7
            expected_steps = [0, 1, 2, 3] if seed else [2, 3]
            assert [v["step"] for v in w["target_counts"]] == expected_steps
            assert [v["shifted"] for v in w["target_counts"]] == [
                ([49, 57, 57, 21][i]) for i in expected_steps
            ]
            assert [v["raw_unshifted"] for v in w["target_counts"]] == [
                ([51, 60, 59, 23][i]) for i in expected_steps
            ]
            dtype = {
                "fp32": "torch.float32",
                "bf16": "torch.bfloat16",
                "fp16": "torch.float16",
                "overflow": "torch.float16",
            }[case]
            obs = w["observed_dtypes"]
            assert obs["linear_outputs"] == obs["forward_weights"] == [dtype]
            assert (
                obs["reduction_dtype"] == "torch.float32"
                and obs["reduction_calls"] == len(expected_steps) * 3
            )
            assert obs["reduction_input_bytes"] == len(expected_steps) * 85120
            assert obs["scaler"] == (
                None if case in ["fp32", "bf16"] else "ShardedGradScaler"
            )
            gathers = w["parameter_allgathers"]
            assert len(gathers) == (35 if seed else 15)
            assert all(
                g["dtype"] == dtype
                and g["input_bytes"] == g["elements"] * (4 if case == "fp32" else 2)
                for g in gathers
            )
            for error in w.get("gradient_errors", []):
                assert error["CPU_AMP"] <= (0.02 if case == "bf16" else 0.004)
                assert error["CPU_FP64"] <= (0.03 if case == "bf16" else 0.006)
            for clip in w["clip_records"]:
                assert 0 < clip["coefficient"] < 1
                assert math.isclose(
                    clip["norm"], clip["CPU_norm"], rel_tol=1e-5, abs_tol=1e-7
                )
            if case in ["fp16", "overflow"]:
                assert obs["final_scaler"]["scale"] == (512 if case == "fp16" else 128)
                assert obs["final_scaler"]["_growth_tracker"] == (
                    0 if case == "fp16" else 1
                )
            if case == "overflow":
                assert w["raw_rank_nonfinite_flags"] == [False, True]
            if not seed:
                old = h["records"][case + "-seed"]["worker"]
                assert (
                    w["birth"]["uuid"] != old["birth"]["uuid"]
                    and w["birth"]["start_ticks"] != old["birth"]["start_ticks"]
                )
                assert (
                    f["container_id"]
                    != h["records"][case + "-seed"]["final"]["container_id"]
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
        for label, files in h["payload_files"].items():
            assert len(files) == 6
            recorded = h["records"][label]["worker"]["checkpoint_files"]
            assert all(
                files[i["name"]]["sha256"] == i["sha256"]
                for i in recorded
                if i["name"] in files
            )
        return h, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host104", type=Path, required=True)
    p.add_argument("--host105", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    left, lg = read(a.host104)
    right, rg = read(a.host105)
    for label in PHASES:
        lw = left["records"][label]["worker"]
        rw = right["records"][label]["worker"]
        assert lw["rank"] == 0 and rw["rank"] == 1
        if label == "bert-control-seed":
            continue
        for l, r in zip(lw["target_counts"], rw["target_counts"]):
            assert (
                sum(l["local_active"]) + sum(r["local_active"])
                == l["shifted"]
                == r["shifted"]
            )
        case = label.split("-")[0]
        resume = label.endswith("resume")
        assert len(lw["checks"]) == (
            (11 if case == "overflow" else 9) if resume else 19
        )
        assert len(rw["checks"]) == (
            (11 if case == "overflow" else 9) if resume else 18
        )
    archives = [
        dict(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
        )
        for path in [a.host104, a.host105]
    ]
    errors = {
        case: {
            key: max(
                e[key]
                for h in [left, right]
                for e in h["records"][case + "-seed"]["worker"]["gradient_errors"]
            )
            for key in ["CPU_AMP", "CPU_FP64"]
        }
        for case in ["bf16", "fp16", "overflow"]
    }
    value = dict(
        status="passed",
        groups=lg + rg,
        guards=18,
        all_guards_acknowledged=True,
        CPU_contract_patterns=15,
        CPU_contract_observations=256,
        parameters=21280,
        parameter_and_AdamW_elements_per_rank=10640,
        target_counts=[49, 57, 57, 21],
        gradient_errors=errors,
        payload_bytes={
            case: sum(x["bytes"] for x in files.values())
            for case, files in left["payload_files"].items()
        },
        archives=archives,
        boots=[left["boot_id"], right["boot_id"]],
        native_runtime_changed=False,
        precision_policies=[
            "FP32",
            "BF16 working parameters",
            "FP16 working parameters/native ShardedGradScaler",
        ],
        next="PLAN-04: explicit frozen/PEFT classic FSDP admission, then activation-checkpointing combinations; retain refusal until bounded proof",
    )
    a.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
