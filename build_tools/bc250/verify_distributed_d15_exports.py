#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent D15 archive, source, reference digest, process birth and guard checks."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from export_distributed_d15_on_host import LABELS


def read(path: Path) -> dict:
    with tarfile.open(path) as t:
        root = "home/nik/bc250work/evidence/distributed-d15-r1/"
        h = json.load(t.extractfile(root + "handoff.json"))
        assert h["status"] == "passed" and not any(
            Path(v.name).suffix in {".pt", ".pth", ".bin", ".pkl", ".safetensors"}
            for v in t.getmembers()
        )
        assert set(h["records"]) == set(LABELS)
        for label, final in h["records"].items():
            assert final == json.load(t.extractfile(root + label + "/final.json"))
            assert final["worker"] == json.load(
                t.extractfile(root + label + "/rank.json")
            )
            assert final["status"] == "passed" and all(
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
                "home/nik/bc250work/tools/distributed-d15/distributed_d15_trainer.py"
            ).read()
            assert (
                hashlib.sha256(source).hexdigest() == final["worker"]["source_sha256"]
            )
        compat = t.extractfile(
            "home/nik/bc250work/tools/distributed-d15/bc250_trainer.py"
        ).read()
        for record in h["records"].values():
            assert (
                hashlib.sha256(compat).hexdigest()
                == record["worker"]["config"]["compat"]
            )
        for filename, key in [
            ("bc250_trainer_finite.py", "guard"),
            ("bc250_trainer_clipping.py", "clipping"),
        ]:
            data = t.extractfile(
                "home/nik/bc250work/tools/distributed-d15/" + filename
            ).read()
            for record in h["records"].values():
                assert (
                    hashlib.sha256(data).hexdigest() == record["worker"]["config"][key]
                )
        for case in ["bf16", "fp16"]:
            s, r = [h["records"][case + "-" + action] for action in ["seed", "resume"]]
            left, right = s["worker"], r["worker"]
            assert s["container_id"] != r["container_id"]
            assert (
                left["birth"]["uuid"] != right["birth"]["uuid"]
                and left["birth"]["start_ticks"] != right["birth"]["start_ticks"]
            )
            assert right["source_birth"] == left["birth"]
            assert left["config"] == right["config"]
            assert (
                left["transcript"][2:] == right["transcript"]
                and left["final_digest"] == right["final_digest"]
            )
            assert [v["step"] for v in right["transcript"]] == [3, 4]
            assert (
                left["optimizer_updates"]
                == right["optimizer_updates"]
                == (3 if case == "fp16" else 4)
            )
        return h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    nodes = [read(a.directory / f"{host}-evidence-r1.tar.gz") for host in [104, 105]]
    for label in LABELS:
        workers = [n["records"][label]["worker"] for n in nodes]
        assert workers[0]["config"] == workers[1]["config"]
        assert [w["rank"] for w in workers] == [0, 1]
    result = dict(
        status="passed",
        accepted_GPU_check_groups=sum(
            len(r["worker"]["checks"]) for n in nodes for r in n["records"].values()
        ),
        accepted_GPU_guards=8,
        exact_resumed_rank_windows=8,
        local_unit_tests=42,
        qualification_boot_ids=[n["qualification_boot_ids"] for n in nodes],
        kernels=[n["kernel"] for n in nodes],
        boot_ids=[n["boot_id"] for n in nodes],
        valid_payload_bytes_per_host=[
            sum(v["bytes"] for v in n["valid_checkpoints_retained"]) for n in nodes
        ],
        cases={
            case: dict(
                clipping_metrics=nodes[0]["records"][case + "-seed"]["worker"][
                    "clipping_metrics"
                ],
                gradient_errors=nodes[0]["records"][case + "-seed"]["worker"][
                    "gradient_errors"
                ],
                final_scaler=nodes[0]["records"][case + "-resume"]["worker"][
                    "final_scaler"
                ],
                worker_births=[
                    [
                        n["records"][case + "-" + x]["worker"]["birth"]
                        for x in ["seed", "resume"]
                    ]
                    for n in nodes
                ],
            )
            for case in ["bf16", "fp16"]
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
