#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export bounded replay metadata or verify fresh-process checkpoint replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export(output):
    assert os.geteuid() == 0, "Complete recovery records require root"
    home = Path("/home/nik")
    evidence = home / "bc250work/evidence/lora-resume"
    project = home / "bc250-projects/lora-smoke/project"
    selected = {}

    def add(path, name):
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size < 8 * 1024**2
        selected[name] = path

    for path in evidence.rglob("*"):
        if path.is_file() and path.suffix in [".json", ".log", ".txt"]:
            add(path, "evidence/" + str(path.relative_to(evidence)))
    for path in (project / "outputs/two-host-tests").glob("l02-*"):
        for file in path.rglob("*"):
            if file.is_file() and (
                file.name.startswith(("replay-rank", "audit-rank"))
                or file.name
                in ["run_config.json", "metrics.jsonl", "train.log", "bc250-l02.json"]
            ):
                add(file, "application/" + str(file.relative_to(project)))
    for file in Path(__file__).parent.iterdir():
        if file.is_file() and file.suffix in [".py", ".json"]:
            add(file, "tools/" + file.name)
    for name in [
        "train_two_host_l02.py",
        "bc250_lora_checkpoint.py",
        "checkpoint_files.py",
        "run-distributed-l02",
    ]:
        add(project / name, "application/" + name)
    inventory = {
        name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
        for name, p in selected.items()
    }
    import io

    with tarfile.open(output, "x:gz") as t:
        for name, path in sorted(selected.items()):
            t.add(path, arcname=name, recursive=False)
        data = json.dumps(inventory, indent=2).encode()
        info = tarfile.TarInfo("inventory.json")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    return {
        "files": len(selected) + 1,
        "bytes": output.stat().st_size,
        "sha256": sha(output.read_bytes()),
    }


def load(path):
    result = {}
    with tarfile.open(path) as t:
        for m in t:
            assert (
                m.isfile()
                and not PurePosixPath(m.name).is_absolute()
                and ".." not in PurePosixPath(m.name).parts
            )
            assert m.name not in result
            result[m.name] = t.extractfile(m).read()
    inventory = json.loads(result.pop("inventory.json"))
    assert set(result) == set(inventory)
    for name, data in result.items():
        assert inventory[name] == dict(bytes=len(data), sha256=sha(data))
    return result


def verify(directory):
    nodes = {host: load(directory / f"{host}-evidence.tar.gz") for host in [104, 105]}
    accepted = []
    failed = []
    manifests = []
    records = []
    for host, files in nodes.items():
        for name, data in files.items():
            if name.startswith("evidence/") and name.endswith("/final.json"):
                r = json.loads(data)
                assert r["registry_unchanged"] and not r["foreign_containers"]
                if not r["kernel_log_unchanged"]:
                    assert host == 105 and r["config"]["label"] == "l02-r2-resume"
                    assert r["status"] == "failed" and r["exit_code"] == 0
                    before = files[
                        name.removesuffix("final.json") + "kernel-before.log"
                    ]
                    after = files[name.removesuffix("final.json") + "kernel-after.log"]
                    assert after.startswith(before)
                    messages = [
                        x.split("] ", 1)[1]
                        for x in after[len(before) :].decode().splitlines()
                    ]
                    assert messages and all(
                        x.startswith(
                            ("eth0: renamed", "veth", "docker0:", "systemd-journald[")
                        )
                        for x in messages
                    )
                prefix = name.removesuffix("final.json")
                state = json.loads(files[prefix + "guard/state.json"])
                ack = json.loads(files[prefix + "guard/ack.json"])
                assert state["status"] == "acknowledged" and not state["dry_run"]
                assert all(state[k] == ack[k] for k in ["nonce", "boot_id", "phase"])
                item = {
                    "host": host,
                    "phase": r["config"]["label"],
                    "pid": r["guardian_pid"],
                    "boot_id": r["boot_id"],
                    "kernel_log_unchanged": r["kernel_log_unchanged"],
                }
                if r["config"]["label"] == "l02-r3-resume":
                    assert r["status"] == "passed" and r["kernel_log_unchanged"]
                    assert r["docker_containers"] == {"before": [], "after": []}
                (accepted if r["status"] == "passed" else failed).append(item)
        rank = host - 104
        root = "application/outputs/two-host-tests/l02-r2/"
        seed = json.loads(files[root + f"seed/replay-rank{rank}.json"])
        resumed = json.loads(files[root + f"resume-r3/replay-rank{rank}.json"])
        manifest = json.loads(files[root + "seed/checkpoint-2/bc250-l02.json"])
        assert seed["status"] == resumed["status"] == "passed"
        assert len(seed["steps"]) == 4 and len(resumed["steps"]) == 2
        assert seed["steps"][2:] == resumed["steps"]
        assert seed["instance_uuid"] != resumed["instance_uuid"]
        assert seed["start_ticks"] != resumed["start_ticks"]
        saved = manifest["ranks"][str(rank)]["state"]
        for key in [
            "step",
            "parameters",
            "optimizer",
            "optimizer_steps",
            "optimizer_state_entries",
            "scaler",
            "scheduler",
            "trl_total_train_tokens",
        ]:
            assert resumed["initial"][key] == saved[key], key
        assert resumed["steps"][0]["before"]["rng"] == saved["rng"]
        assert saved["scaler_state"]["scale"] == 512
        assert (
            saved["optimizer_steps"] == [2] and saved["optimizer_state_entries"] == 372
        )
        assert [s["optimizer_skipped"] for s in seed["steps"]] == [
            False,
            False,
            False,
            True,
        ]
        last = seed["steps"][-1]
        assert not last["finite_gradients"]
        assert (
            last["before"]["scaler_state"]["scale"] == 1024
            and last["after"]["scaler_state"]["scale"] == 512
        )
        for k in ["parameters", "optimizer", "scheduler"]:
            assert last["before"][k] == last["after"][k]
        records.append(
            {
                "host": host,
                "seed_pid": seed["pid"],
                "resume_pid": resumed["pid"],
                "hostname": seed["container_hostname"],
                "seed_process_start_ticks": seed["start_ticks"],
                "resume_process_start_ticks": resumed["start_ticks"],
                "replayed_steps": [s["step"] for s in resumed["steps"]],
                "all_successful_gradients_and_full_state_exact": True,
                "native_overflow_skip_replayed_exactly": True,
                "restored_token_counter": saved["trl_total_train_tokens"],
            }
        )
        manifests.append(manifest)
    assert manifests[0] == manifests[1]
    assert len(accepted) == 5 and len(failed) == 3
    negative = json.loads(nodes[105]["evidence/actual-negative-controls.json"])
    assert len(negative) == 3 and all(negative.values())
    return {
        "status": "passed",
        "accepted_guards": accepted,
        "retained_failed_guards": failed,
        "fresh_process_replays": records,
        "checkpoint_id": manifests[0]["checkpoint_id"],
        "common_payload_files": len(manifests[0]["payload"]),
        "common_payload_bytes": sum(
            v["bytes"] for v in manifests[0]["payload"].values()
        ),
        "local_unit_tests": 54,
        "actual_checkpoint_negative_controls": negative,
        "limits": [
            "Same two ranks, pinned framework/provider, same settings/data/model",
            "Middle-epoch complete optimizer boundary; workers0/unpacked/native AdamW",
            "No elastic or one-host conversion, partial-window or crash-durability proof",
            "No throughput or full-epoch convergence claim; native aggregate train_loss caveat retained",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "verify"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = export(args.path) if args.action == "export" else verify(args.path)
    if args.action == "verify":
        (args.path / "verified-results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result, indent=2))
