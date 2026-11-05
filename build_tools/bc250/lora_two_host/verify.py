#!/usr/bin/env python3
"""Verify owned metadata exports and compare equal-data application timings."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def load(path):
    result = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name not in result
            result[member.name] = archive.extractfile(member).read()
    inventory = json.loads(result["export-inventory.json"])
    assert set(result) == set(inventory) | {"export-inventory.json"}
    for name, meta in inventory.items():
        assert len(result[name]) == meta["bytes"]
        assert hashlib.sha256(result[name]).hexdigest() == meta["sha256"]
    return result


def verify(directory):
    nodes = {h: load(directory / f"{h}-evidence.tar.gz") for h in [104, 105]}
    for h, files in nodes.items():
        old = json.loads(files["tools/source-inventory-r1.json"])
        current = json.loads(files["tools/source-inventory-r2.json"])
        assert old["104"] == old["105"] and current["104"] == current["105"]
        for name in ["train_two_host.py", "bc250_lora_support.py", "run-distributed"]:
            assert (
                hashlib.sha256(files["application/" + name]).hexdigest()
                == current[str(h)]["project/" + name]
            )
        assert (
            hashlib.sha256(files["tools/support-r1.py"]).hexdigest()
            == old[str(h)]["project/bc250_lora_support.py"]
        )
        assert (
            hashlib.sha256(
                files["application/train_two_host.46da42893ae5.py"]
            ).hexdigest()
            == old[str(h)]["project/train_two_host.py"]
        )
        for name in [
            "transformers/trainer.py",
            "trl/trainer/sft_trainer.py",
            "accelerate/accelerator.py",
        ]:
            assert (
                hashlib.sha256(files["framework/" + name]).hexdigest()
                == current[str(h)]["env/lib/python3.12/site-packages/" + name]
            )
    guards = []
    for h, files in nodes.items():
        for name, content in files.items():
            if name.startswith("evidence/") and name.endswith("/final.json"):
                record = json.loads(content)
                assert record["status"] == "passed"
                assert record["exit_code"] == 0
                assert record["kernel_log_unchanged"] and record["registry_unchanged"]
                assert not record["foreign_containers"]
                prefix = name.removesuffix("final.json")
                state = json.loads(files[prefix + "guard/state.json"])
                ack = json.loads(files[prefix + "guard/ack.json"])
                assert all(state[k] == ack[k] for k in ["nonce", "boot_id", "phase"])
                assert record["boot_id"] == state["boot_id"]
                guards.append(
                    {
                        "host": h,
                        "phase": record["config"]["label"],
                        "guardian_pid": record["guardian_pid"],
                        "boot_id": state["boot_id"],
                    }
                )
    assert len(guards) == 8, len(guards)

    def audit(host, phase):
        rank = host - 104
        name = f"application/outputs/two-host-tests/{phase}/audit-rank{rank}.json"
        r = json.loads(nodes[host][name])
        assert r["device"] == "gfx1013:xnack-" and r["accelerator_accumulation"] == 1
        assert r["torch_path"] == "/workspace/pytorch-work/torch/__init__.py"
        assert r["trainable_parameters"] == 134553600
        assert r["initial_adapter_sha256"] != r["final_adapter_sha256"]
        for row in r["rows"]:
            assert row["finite_gradients"] and not row["optimizer_skipped"]
            assert row["examples"] == 16
        return r

    records = {}
    for series in ["correctness-r1", "timing-r2"]:
        single = audit(104, "single-" + series)
        pair = [audit(h, "ddp-" + series) for h in [104, 105]]
        assert (
            single["initial_adapter_sha256"]
            == pair[0]["initial_adapter_sha256"]
            == pair[1]["initial_adapter_sha256"]
        )
        assert pair[0]["final_adapter_sha256"] == pair[1]["final_adapter_sha256"]
        count = 4 if series == "correctness-r1" else 6
        assert all(len(r["rows"]) == count for r in [single, *pair])
        for a, b, c in zip(single["rows"], pair[0]["rows"], pair[1]["rows"]):
            for k in ["step", "examples", "targets", "input_tokens", "padded_tokens"]:
                assert a[k] == b[k] == c[k], (series, k)
            if series == "timing-r2":
                assert len(a["example_sha256"]) == 16
                assert len(b["example_sha256"]) == len(c["example_sha256"]) == 8
                assert sorted(a["example_sha256"]) == sorted(
                    b["example_sha256"] + c["example_sha256"]
                )
        # Exclude the first update: initialization, allocator warm-up and the
        # optional numerical dump make it unsuitable as a steady-state estimate.
        seconds_single = sum(row["seconds"] for row in single["rows"][1:])
        seconds_pair = sum(
            max(a["seconds"], b["seconds"])
            for a, b in zip(pair[0]["rows"][1:], pair[1]["rows"][1:])
        )
        records[series] = {
            "measured_updates": count - 1,
            "single_seconds": seconds_single,
            "two_host_seconds": seconds_pair,
            "throughput_ratio": seconds_single / seconds_pair,
            "single_examples_per_second": 16 * (count - 1) / seconds_single,
            "two_host_examples_per_second": 16 * (count - 1) / seconds_pair,
            "single_peak_allocated_bytes": max(
                row["peak_allocated_bytes"] for row in single["rows"]
            ),
            "two_host_peak_allocated_bytes": [
                max(row["peak_allocated_bytes"] for row in r["rows"]) for r in pair
            ],
            "per_update_seconds": {
                "single": [row["seconds"] for row in single["rows"]],
                "two_host_maximum": [
                    max(a["seconds"], b["seconds"])
                    for a, b in zip(pair[0]["rows"], pair[1]["rows"])
                ],
            },
        }
    for h in [104, 105]:
        audit(h, "ddp-production-r2")
        for phase in ["ddp-production-r2", "ddp-timing-r2"]:
            text = nodes[h][f"evidence/{phase}/gpu.log"].decode()
            assert "destroy_process_group() was not called" not in text
    saved = json.loads(
        nodes[104][
            "application/outputs/two-host-tests/ddp-production-r2/final_adapter/bc250-save-verification.json"
        ]
    )
    assert saved["status"] == "passed" and saved["elements"] == 134553600
    assert saved["all_saved_adapter_weights_equal_live_weights"]
    gradient = json.loads(nodes[104]["evidence/gradient-comparison-r1.json"])
    assert gradient["status"] == "passed" and gradient["relative_l2"] <= 0.03
    assert gradient["maximum_per_parameter_bound_fraction"] <= 1
    return {
        "status": "passed",
        "guards": guards,
        "gradient_comparison": gradient,
        "saved_adapter": saved,
        "timings": records,
        "input_hash_matches": 96,
        "limits": [
            "Subset, not whole-job speed or convergence proof",
            "One-GPU same-graph gradient reference, not independent CPU model",
            "Separate-disk full application checkpoint restart unqualified",
            "FP32 DDP gradient payload 538214400 bytes/update; not measured wire bytes",
            "Native reference conv/delta paths and recovered allocation retries retained",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    path = args.directory / "verified-results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
