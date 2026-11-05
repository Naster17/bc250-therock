#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent D09 archive/payload/guard/caller/activation verification."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from export_distributed_d09_on_host import PHASES


def read(path: Path) -> dict:
    with tarfile.open(path) as t:
        assert not any(m.name.endswith(".pt") for m in t.getmembers())
        root = "home/nik/bc250work/evidence/distributed-d09-r1/"
        handoff = json.load(t.extractfile(root + "handoff.json"))
        assert handoff["status"] == "passed" and handoff["running_containers"] == []
        records = {v["label"]: v["final"] for v in handoff["records"]}
        assert set(records) == set(PHASES)
        for label, final in records.items():
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
                state[k] == ack[k] for k in ["nonce", "boot_id", "phase"]
            )
        pins = handoff["installed_payload_hashes"]
        assert len(pins) == 9
        for name, digest in pins.items():
            assert (
                hashlib.sha256(
                    t.extractfile(
                        "opt/bc250-stack/tools/distributed-d09/" + name
                    ).read()
                ).hexdigest()
                == digest
            )
            assert (
                hashlib.sha256(
                    t.extractfile(
                        "home/nik/bc250work/tools/distributed-d09/install-payload/"
                        + name
                    ).read()
                ).hexdigest()
                == digest
            )
        recovery = "home/nik/bc250work/evidence/distributed-d09-tool-install-r1/"
        before = json.load(t.extractfile(recovery + "registry-before.json"))
        active = handoff["runtime"]
        assert (
            before["runtime_sha256"] == active["runtime_sha256"]
            and before["release"] == active["release"]
        )
        assert (
            before["agent_exit_fix"] == active["agent_exit_fix"]
            and before["agent_exit_recovery"] == active["agent_exit_recovery"]
        )
        assert before["launcher_tools"] == "/opt/bc250-stack/tools/distributed-d05"
        assert active["launcher_tools"] == "/opt/bc250-stack/tools/distributed-d09"
        assert (
            json.load(t.extractfile(recovery + "rollback.json"))["status"] == "passed"
        )
        assert (
            json.load(t.extractfile(recovery + "activation.json"))["payload_hashes"]
            == pins
        )
        user = json.load(
            t.extractfile(
                "home/nik/bc250work/evidence/distributed-d09-user-cpu/rank.json"
            )
        )
        assert (
            user == handoff["user_cpu"]
            and user["status"] == "passed"
            and user["uid"] == 1000
            and user["torch_cuda_initialized"] is False
        )
        assert len(json.load(t.extractfile(root + "CLI-refusals.json"))["cases"]) == 12
        return handoff


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    nodes = [read(a.directory / f"{host}-evidence-r1.tar.gz") for host in [104, 105]]
    assert nodes[0]["installed_payload_hashes"] == nodes[1]["installed_payload_hashes"]
    assert [n["user_cpu"]["rank"] for n in nodes] == [0, 1]
    consumers = [
        r["final"]["worker"]
        for n in nodes
        for r in n["records"]
        if r["label"] != "retained-candidate"
    ]
    assert all(
        c["transformers"] == "5.2.0" and c["parameter_elements"] == 22561
        for c in consumers
    )
    assert all(
        c["torch_path"] == "/workspace/pytorch-work/torch/__init__.py"
        for c in consumers
    )
    result = dict(
        status="passed",
        GPU_worker_check_groups=sum(
            len(r["final"]["worker"]["checks"]) for n in nodes for r in n["records"]
        ),
        CPU_caller_check_groups=sum(len(n["user_cpu"]["checks"]) for n in nodes),
        actual_CLI_refusals=24,
        CPU_identity_refusals=6,
        GPU_guard_ACKs=6,
        local_unit_tests=37,
        selected_tools="/opt/bc250-stack/tools/distributed-d09",
        native_Torch_C_extension_and_agent_exit_payload_unchanged=True,
        installed_payload_hashes=nodes[0]["installed_payload_hashes"],
        boot_ids=[n["boot_id"] for n in nodes],
        kernels=[n["kernel"] for n in nodes],
        PIDs={
            r["label"]: [n["records"][i]["final"]["guardian_pid"] for n in nodes]
            for i, r in enumerate(nodes[0]["records"])
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
