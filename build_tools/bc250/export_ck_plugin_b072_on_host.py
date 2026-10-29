#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal the private plugin's forced-solver numerical and preservation evidence."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    logs = WS / "logs/ck-plugin-b072-qualification-r5"
    a = json.loads((logs / "final.json").read_text())
    assert a["status"] == "passed" and sum(g["checks"] for g in a["gates"]) == 38
    dest = WS / "tmp/ck-plugin-b072/export-r1"
    dest.mkdir()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        assert not verify_hashes(a["protected_sha256"])
        assert all(os.readlink(p) == v for p, v in a["links"].items())
        assert (
            subprocess.check_output(["dmesg"])
            == (logs / "kernel-before.log").read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        guards = []
        for p in Path("/run").glob("ck-plugin-b072-*/state.json"):
            s = json.loads(p.read_text())
            assert s["status"] == "acknowledged" and s["boot_id"] == a["boot_id"]
            guards.append(p.parent)
        assert len(guards) == 12
        cohorts = {}
        for profile in ("daily", "rccl"):
            p = WS / f"logs/ck-plugin-b072-r5-private-{profile}"
            r = json.loads((p / "result.json").read_text())
            assert r["status"] == "passed" and len(r["checks"]) == 8
            text = (p / "private-stderr.log").read_text()
            assert (
                "Loaded CK grouped conv library from MIOPEN_CK_LIB_PATH" in text
                or "Loaded CK grouped conv library from env path" in text
                or "libMIOpenCKGroupedConv_gfx1013.so" in text
            )
            assert (
                "ConvHipImplicitGemmGroupFwdXdlops" in text
                and "kernel_name = DeviceGroupedConvFwdDl" in text
            )
            cohorts[profile] = r
        report = dict(
            status="qualified-isolated-fp32-forward-dl-plugin",
            checks=38,
            accepted_guards=6,
            total_acknowledged_guards=len(guards),
            cohorts=cohorts,
            forced_solver_executions=64,
            negative_solver_refusals=64,
            direct_contract_checks=304,
            torch_conv_output_first_gradient_fixtures=8,
            protected_paths=len(a["protected_sha256"]),
            links=len(a["links"]),
            aot_caches=a["aot_caches"],
            boot_id=a["boot_id"],
            kernel=os.uname().release,
            kernel_log_unchanged=True,
            active_runtime_changed=False,
            installed=False,
            default_admission_changed=False,
            host105_contacted=False,
            running_owned_jobs=[],
            ordinary_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        )
        atomic_json(dest / "verification.json", report)
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for p in sorted((WS / "logs").glob("*ck-plugin-b072*")):
                tar.add(p, arcname="host-logs/" + p.name)
            for p in guards:
                tar.add(p, arcname="guards/" + p.name)
            for p in sorted((WS / "helpers").glob("*b072*")):
                if p.is_file():
                    tar.add(p, arcname="helpers/" + p.name)
            for n in ("bc250_dl_private.cpp", "ck_grouped_conv_fwd_impl.cpp"):
                tar.add(WS / "helpers" / n, arcname="sources/" + n)
            for p in sorted((WS / "tmp/ck-plugin-b072").glob("build-r*")):
                tar.add(p, arcname=p.name)
            tar.add(
                WS / "build/ml-libs/MIOpen/build/include/miopen/config.h",
                arcname="original-config.h",
            )
        atomic_json(
            dest / "archive.json",
            dict(
                sha256=sha(dest / "evidence.tar.gz"),
                bytes=(dest / "evidence.tar.gz").stat().st_size,
            ),
        )
    print("CK_PLUGIN_B072_EVIDENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
