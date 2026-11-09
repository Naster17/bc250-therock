#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify guarded D31 BF16 all-adapter numerical references and exact native replay."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from export_distributed_d31_on_host import PHASES


def read(path):
    with tarfile.open(path) as archive:
        prefix = "home/nik/bc250work/evidence/distributed-d31-r1/"
        source = "home/nik/bc250work/tools/distributed-d31/"

        def record(name):
            return json.load(archive.extractfile(prefix + name))

        handoff = record("handoff.json")
        assert handoff["status"] == "passed"
        assert handoff["original_checkpoint_files_unchanged"]
        assert handoff["rank1_only_private_policy_change_verified"]
        assert handoff["checkpoint_policy"]["precision"] == dict(
            working_parameters="bfloat16",
            master_parameters="float32",
            reductions="float32",
            buffers="float32",
            keep_low_precision_grads=False,
            cast_forward_inputs=False,
            cast_root_forward_inputs=False,
            activation_checkpointing=True,
        )
        helper = "bc250_fsdp_peft_bf16_checkpoint.py"
        assert (
            archive.extractfile(source + helper).read()
            == Path(__file__).with_name(helper).read_bytes()
        )
        assert handoff["checkpoint_policy"]["activation_checkpointing"] == dict(
            use_reentrant=False, preserve_rng_state=True
        )
        for dependency in ["bc250_fsdp_peft_bf16.py", "bc250_fsdp_activation.py"]:
            assert (
                archive.extractfile(source + dependency).read()
                == Path(__file__).with_name(dependency).read_bytes()
            )
        groups = observations = 0
        for series, label, accepted in PHASES:
            assert series == "r1" and accepted
            w = record(label + "/rank.json")
            f = record(label + "/final.json")
            assert handoff["records"][series + "/" + label] == dict(final=f, worker=w)
            assert w["status"] == f["status"] == "passed" and f["worker"] == w
            assert f["kernel_log_unchanged"] and f["registry_unchanged"]
            assert f["agent_shutdown_clean"] and f["docker_containers"] == dict(
                before=[], after=[]
            )
            assert (
                archive.extractfile(prefix + label + "/kernel-before.log").read()
                == archive.extractfile(prefix + label + "/kernel-after.log").read()
            )
            state = record(label + "/guard/state.json")
            ack = record(label + "/guard/ack.json")
            assert state["status"] == "acknowledged"
            assert all(state[k] == ack[k] for k in ["nonce", "phase", "boot_id"])
            assert state["boot_id"] == handoff["boot_id"]
            groups += len(w["checks"])
            observations += len(w["CPU_policy_contracts"]) + len(
                w["CPU_causal_contracts"]
            )
            fixture = (
                "distributed_d30_trainer.py"
                if label == "retained-control-seed"
                else (
                    "distributed_d31_allocator_control.py"
                    if label == "allocator-control-seed"
                    else "distributed_d31_trainer.py"
                )
            )
            for filename, sha in [
                ("distributed_d25_trainer.py", w["source_sha256"]),
                ("bc250_trainer_fsdp.py", w["config"]["helper_sha256"]),
                (fixture, w["fixture_source_sha256"]),
                ("bc250_fsdp_frozen.py", w["frozen_helper_sha256"]),
                ("bc250_fsdp_peft.py", w["peft_helper_sha256"]),
            ]:
                assert (
                    hashlib.sha256(
                        archive.extractfile(source + filename).read()
                    ).hexdigest()
                    == sha
                )
            assert (
                w["config"]["parameters"] == 24352
                and w["config"]["trainable_parameters"] == 3072
            )
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
                assert (
                    "parameters-optimizer-scheduler-unchanged-after-refusal"
                    in w["checks"]
                )
                continue
            if label == "retained-control-seed":
                assert w["frozen_gate_calls"] == 1
                continue
            seed = label != "frozen-resume"
            windows = 4 if seed else 2
            assert (
                w["optimizer_shard_elements"] == 1536
                and w["frozen_gate_calls"] == windows
            )
            assert [v["shifted"] for v in w["target_counts"]] == (
                [49, 57, 57, 21] if seed else [57, 21]
            )
            assert w["checkpoint_block_calls"] == (
                [7, 7]
                if label == "allocator-control-seed"
                else [14, 14] if seed else [6, 6]
            )
            assert len(w["parameter_allgathers"]) == (
                259 if label == "allocator-control-seed" else 357 if seed else 153
            )
            obs = w["observed_dtypes"]
            assert obs["scaler"] is None and obs["final_scaler"] == {}
            assert obs["linear_outputs"] == obs["forward_weights"] == ["torch.bfloat16"]
            assert (
                obs["reduction_calls"] == windows * 16
                and obs["reduction_input_bytes"] == windows * 12288
            )
            assert all(
                g["dtype"] == "torch.bfloat16" and g["input_bytes"] == g["elements"] * 2
                for g in w["parameter_allgathers"]
            )
            assert all(
                v["parameters_exact"] and v["logits_exact"] and v["rng_unchanged"]
                for v in w["adapter_exports"]
            )
        seed = handoff["records"]["r1/frozen-seed"]["worker"]
        replay = handoff["records"]["r1/frozen-resume"]["worker"]
        baseline = handoff["records"]["r1/allocator-control-seed"]["worker"]
        assert seed["transcript"] == baseline["transcript"]
        assert seed["final_state_digest"] == baseline["final_state_digest"]
        assert (
            replay["birth"]["uuid"] != seed["birth"]["uuid"]
            and replay["birth"]["start_ticks"] != seed["birth"]["start_ticks"]
        )
        assert (
            handoff["records"]["r1/frozen-resume"]["final"]["container_id"]
            != handoff["records"]["r1/frozen-seed"]["final"]["container_id"]
        )
        assert replay["transcript"] == [v for v in seed["transcript"] if v["step"] > 2]
        assert seed["final_state_digest"] == replay["final_state_digest"]
        for value in seed["gradient_errors"]:
            assert value["CPU_AMP"] <= 0.02 and value["CPU_FP64"] <= 0.03
        assert len(seed["gradient_errors"]) == 4
        assert (
            replay["restore_optimizer_actual_keys"]
            == replay["restore_optimizer_expected_keys"]
            and replay["restore_optimizer_actual_keys"]
        )
        assert len(handoff["payload_files"]) == 6
        return handoff, groups, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host104", type=Path, required=True)
    parser.add_argument("--host105", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline104", type=Path, required=True)
    parser.add_argument("--baseline105", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    args = parser.parse_args()
    left, lg, lo = read(args.host104)
    right, rg, ro = read(args.host105)
    assert left["checkpoint_policy"] == right["checkpoint_policy"]
    assert left["payload_files"] == right["payload_files"]
    baseline_proof = json.loads(args.baseline_results.read_text())
    assert (
        baseline_proof["status"] == "passed"
        and not baseline_proof["activation_checkpointing"]
    )
    baseline_archives = []
    for baseline, current, published in zip(
        [args.baseline104, args.baseline105], [left, right], baseline_proof["archives"]
    ):
        sha = hashlib.sha256(baseline.read_bytes()).hexdigest()
        assert sha == published["sha256"]
        with tarfile.open(baseline) as archive:
            before = json.load(
                archive.extractfile(
                    "home/nik/bc250work/evidence/distributed-d30-r1/frozen-seed/rank.json"
                )
            )
        after = current["records"]["r1/frozen-seed"]["worker"]
        assert before["transcript"] == after["transcript"]
        assert before["final_state_digest"] == after["final_state_digest"]
        baseline_archives.append(dict(path=str(baseline), sha256=sha))
    result = dict(
        status="passed",
        groups=lg + rg,
        guards=18,
        CPU_contract_observations=lo + ro,
        parameters=24352,
        frozen=21280,
        trainable=3072,
        full_exact_restart=True,
        activation_checkpointing=True,
        native_runtime_changed=False,
        checkpointing_off_transcript_exact=True,
        qualified_D30_baseline_archives=baseline_archives,
        BF16_gradient_max_relative=dict(
            CPU_AMP=max(
                v["CPU_AMP"]
                for h in [left, right]
                for v in h["records"]["r1/frozen-seed"]["worker"]["gradient_errors"]
            ),
            CPU_FP64=max(
                v["CPU_FP64"]
                for h in [left, right]
                for v in h["records"]["r1/frozen-seed"]["worker"]["gradient_errors"]
            ),
        ),
        BF16_helper_sha256=hashlib.sha256(
            Path(__file__).with_name("bc250_fsdp_peft_bf16_checkpoint.py").read_bytes()
        ).hexdigest(),
        allocator=[
            dict(
                rank=rank,
                checkpointed_peak_bytes=h["records"]["r1/frozen-seed"]["worker"][
                    "allocator_peak_bytes"
                ],
                checkpointing_off_peak_bytes=h["records"]["r1/allocator-control-seed"][
                    "worker"
                ]["allocator_peak_bytes"],
                scope="tiny complete correctness harness; no capacity/speed proof",
            )
            for rank, h in enumerate([left, right])
        ],
        payload_bytes=sum(v["bytes"] for v in left["payload_files"].values()),
        boot_ids=[left["boot_id"], right["boot_id"]],
        checkpoint_policy=left["checkpoint_policy"],
        archives=[
            dict(
                path=str(p),
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
            for p in [args.host104, args.host105]
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "checkpoint_policy"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
