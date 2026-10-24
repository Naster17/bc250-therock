#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export bounded W8 audit/qualification evidence without copying native builds."""

import json
import os
from pathlib import Path
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_quant_b051_on_host import SOURCES
from qualify_training_n33_on_host import WS


def main():
    assert os.geteuid() == 0
    report = json.loads(
        (WS / "logs/quant-b051-qualification-r1/final.json").read_text()
    )
    assert report["status"] == "passed"
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in report["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    first = WS / "logs/quant-b051-audit-r1"
    failed = json.loads((first / "result.json").read_text())
    assert failed["status"] == "failed" and "derivative" in failed["error"]
    before = (first / "kernel-before.log").read_bytes()
    assert subprocess.check_output(["dmesg"]) == before
    guards = []
    for p in sorted(Path("/run").glob("quant-b051-*")):
        if not (p / "state.json").is_file():
            continue
        state = json.loads((p / "state.json").read_text())
        assert state["status"] == "acknowledged"
        guards.append(dict(label=p.name, status=state["status"]))
    assert len(guards) == 7
    destination = WS / "quant-b051"
    destination.mkdir()
    audit = json.loads((WS / "logs/quant-b051-audit-r2/result.json").read_text())
    prototype = json.loads(
        (WS / "logs/quant-b051-prototype-r1/result.json").read_text()
    )
    trace = json.loads(
        (WS / "logs/quant-b051-prototype-r1/quant-native-trace.json").read_text()
    )
    kernels = [
        e["name"]
        for e in trace["traceEvents"]
        if e.get("cat") == "kernel" and "weight_int8pack_mm_kernel" in e.get("name", "")
    ]
    assert len(kernels) == 2, kernels
    sources = (*SOURCES, "export_quant_b051_on_host.py")
    summary = dict(
        status="passed",
        checks=sum(g["checks"] for g in report["gates"]),
        gates=len(report["gates"]),
        protected_paths=len(report["protected_sha256"]),
        changed_hashes=report["changed_hashes"],
        links_unchanged=report["links_unchanged"],
        aot_caches=report["aot_caches"],
        source_sha256={n: sha(WS / "helpers" / n) for n in sources},
        torch_source_sha256=report["source_provenance_sha256"],
        native_torch_revision=report["native_torch_revision"],
        packages=audit["packages"],
        triton=audit["triton"],
        torch=audit["torch"],
        hip=audit["hip"],
        native_findings=audit["checks"],
        prototype_checks=len(prototype["checks"]),
        prototype_timings=prototype["timings"],
        kernel_names=kernels,
        guards=guards,
        boot_id=report["boot_id"],
        kernel=os.uname().release,
        kernel_log_unchanged_including_failed_probe=True,
        daily_runtime_sha256=sha(Path("/opt/bc250-stack/current/runtime.json")),
        opt_in_runtime_sha256=sha(
            Path("/opt/bc250-stack/profiles/rccl-single/runtime.json")
        ),
        native_build_required=False,
        default_changed=False,
        installed=False,
        host105_contacted=False,
        running_jobs=[],
        running_containers=[],
    )
    assert summary["checks"] == 110 and summary["gates"] == 6
    atomic_json(destination / "verification.json", summary)
    archive = destination / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted((WS / "logs").glob("quant-b051-*")):
            tar.add(p, arcname="logs/" + p.name)
        for n in (*sources, "audit_quant_b051-r1.py"):
            tar.add(WS / "helpers" / n, arcname="helpers/" + n)
        source = Path("/opt/bc250-stack/current/torch")
        for n in report["source_provenance_sha256"]:
            tar.add(source / n, arcname="native-source/" + n)
        for row in guards:
            tar.add(Path("/run") / row["label"], arcname="guards/" + row["label"])
        tar.add(destination / "verification.json", arcname="verification.json")
    atomic_json(
        destination / "archive.json",
        dict(sha256=sha(archive), bytes=archive.stat().st_size),
    )
    for p in destination.iterdir():
        os.chown(p, 1000, 1000)
    print("QUANT_B051_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
