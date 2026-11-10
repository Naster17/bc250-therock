#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read the D44 archive without extraction and replay its stated acceptance gates."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def verify(path):
    with tarfile.open(path) as archive:

        def data(name):
            return archive.extractfile(name).read()

        def record(name):
            return json.loads(data(name))

        root = "evidence/visibility-d44-r2/"
        result = record(root + "result.json")
        assert result["status"] == "passed"
        assert result["guard"]["status"] == "acknowledged"
        state = record(root + "guard/state.json")
        ack = record(root + "guard/ack.json")
        assert state == result["guard"]
        assert all(ack[k] == state[k] for k in ["boot_id", "phase", "nonce"])
        for name, digest in result["sources"].items():
            assert (
                hashlib.sha256(data("tools/visibility-d44/" + name)).hexdigest()
                == digest
            ), name
        phases = result["phases"]
        assert [p["name"] for p in phases] == [
            "old-negative",
            "candidate-negative",
            "candidate-cpu",
            "candidate-daily-default",
            "candidate-daily-zero",
            "candidate-rccl-zero",
        ]
        assert phases[0]["exit"] == 139
        baseline = data(root + "old-negative.log").decode()
        assert (
            "agent.cpp:216" in baseline
            and "Fatal Python error: Segmentation fault" in baseline
        )
        assert "Sequence '-1' in HIP_VISIBLE_DEVICES=-1 not recognized" in baseline
        for phase in phases[1:]:
            label = phase["name"]
            assert phase["exit"] == (2 if label == "candidate-negative" else 0)
            assert phase["kernel_unchanged"]
            assert data(root + label + "-kernel-before.log") == data(
                root + label + "-kernel-after.log"
            )
            assert all(
                s["available_bytes"] >= 3 * 1024**3
                and s["disk_free_bytes"] >= 5 * 1024**3
                for s in phase["samples"]
            )
            if label == "candidate-negative":
                assert not phase["containers"]
                assert (
                    "recorded ROCr import crash" in data(root + label + ".log").decode()
                )
            elif label == "candidate-cpu":
                assert (
                    "CPU_IMPORT_UNINITIALIZED" in data(root + label + ".log").decode()
                )
            else:
                worker = json.loads(
                    data(root + label + ".log").decode().splitlines()[-1]
                )
                assert worker == phase["worker"] and worker["checks"] == 4
                assert (
                    worker["status"] == "passed"
                    and worker["architecture"] == "gfx1013:xnack-"
                )
        install = record("evidence/visibility-d44-install-r1/result.json")
        exported = record("evidence/visibility-d44-install-r1/export.json")
        assert install["status"] == exported["status"] == "passed"
        assert exported["boot_id"] == result["boot_id"]
        assert set(exported["targets"].values()) == {install["source_sha256"]}
        assert install["source_sha256"] == result["sources"]["bc250_torch.py"]
        for item in install["checks"]:
            assert item["exit"] == (1 if item["name"] == "distributed" else 2)
            assert "recorded ROCr import crash" in item["stderr"]
            assert "WORKER_MUST_NOT_RUN" not in item["stderr"]
        assert len(install["checks"]) == 3
        for index in (0, 1):
            assert (
                hashlib.sha256(
                    data(f"evidence/visibility-d44-install-r1/{index}-bc250_torch.py")
                ).hexdigest()
                == install["old_sha256"]
            )
        return dict(
            status="passed",
            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha256=install["source_sha256"],
            gpu_reference_checks=12,
            installed_refusals=3,
            native_parser_fixed=False,
            historical_cpu_fault_fixed=False,
            mixed_stability_qualified=False,
            host="192.168.1.104",
            boot_id=result["boot_id"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.archive), indent=2) + "\n")
