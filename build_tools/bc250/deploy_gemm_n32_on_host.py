#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deploy a qualified exact-layout GEMM preset with a paired loader repair."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()
PREFIX = RELEASE / "rocm"
RECOVERY = WS / "recovery/gemm-n32-before"


def replace(path, data, original=None):
    previous = (original or path).stat() if (original or path).exists() else None
    temp = path.with_name(path.name + ".n32-new")
    assert not temp.exists()
    temp.write_bytes(data)
    temp.chmod(stat.S_IMODE(previous.st_mode) if previous else 0o644)
    if previous:
        os.chown(temp, previous.st_uid, previous.st_gid)
    os.replace(temp, path)


def main():
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    logs = WS / "logs/gemm-n32-deployment-r1"
    logs.mkdir()
    baseline = json.loads(
        (WS / "logs/training-n31-qualification-r4/final.json").read_text()
    )
    old_protected = dict(baseline["protected_sha256"])
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert boot_id == baseline["boot_id"]
    assert os.uname().release == baseline["kernel"]
    for p, target in baseline["links"].items():
        assert str(Path(p).resolve()) == target, p
    for p, d in old_protected.items():
        assert sha(Path(p)) == d, p
    for label in (
        "gemm-n32-layout-r1",
        "gemm-n32-solutions-r1",
        "gemm-n32-hydration-r1",
        "gemm-n32-cold-candidate-r1",
        "gemm-n32-model-candidate-r1",
    ):
        check_gate(label)
        assert (
            json.loads((WS / "logs" / label / "result.json").read_text())["status"]
            == "passed"
        )
    source = json.loads((WS / "helpers/gemm-n32-source.json").read_text())
    for n, d in source["helpers_sha256"].items():
        assert sha(WS / "helpers" / n) == d, n
    candidate = WS / "candidates/gemm-n32-r2/librocblas.so.5.8"
    build = json.loads((WS / "logs/gemm-n32-build-r2/final.json").read_text())
    assert (
        build["status"] == "passed"
        and not build["changed_hashes"]
        and build["kernel_log_unchanged"]
    )
    assert sha(candidate) == build["candidate_sha256"]
    library = PREFIX / "lib/librocblas.so.5.8"
    runtime = RELEASE / "runtime.json"
    launcher = Path("/usr/local/bin/bc250-torch")
    helper = WS / "helpers/bc250_torch.py"
    assert sha(library) == sha(RECOVERY / "librocblas.installed.so.5.8")
    assert sha(runtime) == sha(RECOVERY / "runtime.json")
    assert sha(launcher) == sha(RECOVERY / "bc250-torch")
    (RECOVERY / "bc250_torch.helper.py").write_bytes(helper.read_bytes())
    allowed = {
        p
        for p in old_protected
        if Path(p).resolve() in {library.resolve(), runtime.resolve()}
    }
    identities = {"lib/librocblas.so.5.8": sha(candidate)}
    for p in (PREFIX / "lib/rocblas/library/gfx1013").iterdir():
        if p.is_file():
            identities[str(p.relative_to(PREFIX))] = sha(p)
    assert len(identities) > 100
    csv = WS / "helpers/rocblas_gemm_n32.csv"
    new_csv = RELEASE / "rocblas-gemm-n32.csv"
    assert not new_csv.exists()
    old_runtime = runtime.read_bytes()
    before = (WS / "logs/gemm-n32-layout-r1/kernel-before.log").read_bytes()
    assert subprocess.check_output(["dmesg"]) == before
    report = dict(
        status="installing",
        source=source,
        protected_before=old_protected,
        installed_gates=[],
        candidate_sha256=sha(candidate),
        intentional_mutation_paths=sorted(allowed),
    )
    atomic_json(logs / "before.json", report)
    changed = False
    try:
        replace(library, candidate.read_bytes())
        changed = True
        replace(new_csv, csv.read_bytes())
        updated_launcher = (
            WS / "helpers/gemm-n32-launcher/bc250_torch.py"
        ).read_bytes()
        replace(launcher, updated_launcher)
        replace(helper, updated_launcher)
        manifest = json.loads(old_runtime)
        manifest["previous_n32_rocm_libraries_revision"] = manifest[
            "rocm_libraries_revision"
        ]
        manifest["rocm_libraries_revision"] = source["rocm_libraries_revision"]
        manifest["therock_revision"] = source["therock_code_revision"]
        manifest["rocblas_gemm_tuning"] = dict(
            enabled=True,
            status="installed gates pending",
            sha256=sha(csv),
            identity_sha256=identities,
            problem="BF16 FP32-compute beta0 N,T 2048x1024x6144 lda2048 ldb1024 ldc2048, batch1",
            solution_index=-490,
            disable="BC250_GEMM_TUNING=0",
            limits="Finite synthetic-model/CPU and bounded training-gradient checks; reduction order changes; no sustained throughput/convergence/bitwise checkpoint continuity claim",
        )
        atomic_json(runtime, manifest)
        for suffix, script, extra, count, key in (
            ("cold", "validate_override_gemm_n32.py", [], 3, "checks"),
            ("core", "validate_cleanup_runtime.py", [], 7, "checks"),
            ("conv", "validate_miopen_a01_torch.py", [], 30, "results"),
            ("model", "validate_installed_model_gemm_n32.py", [], 4, "checks"),
            (
                "optout",
                "validate_gemm_optout_n32.py",
                ["--env", "BC250_GEMM_TUNING=0"],
                2,
                "checks",
            ),
        ):
            label = "gemm-n32-installed-" + suffix + "-r1"
            command = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--guard-release",
                os.uname().release,
                "--label",
                label,
                "--script",
                script,
                *extra,
            ]
            with (logs / (suffix + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=570,
                )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed" and len(result[key]) == count
            guard = json.loads((Path("/run") / label / "state.json").read_text())
            assert guard["status"] == "acknowledged" and guard["timeout_seconds"] == 600
            report["installed_gates"].append(
                dict(label=label, checks=count, command=command, guard=guard)
            )
            atomic_json(logs / "progress.json", report)
        unexpected = [
            p
            for p, d in old_protected.items()
            if p not in allowed and sha(Path(p)) != d
        ]
        assert not unexpected, unexpected
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot_id
        for p, target in baseline["links"].items():
            assert str(Path(p).resolve()) == target, p
        assert subprocess.check_output(["dmesg"]) == before
        manifest["rocblas_gemm_tuning"]["status"] = "qualified bounded"
        manifest[
            "source_state"
        ] += "; N3.2 cold override loader repair and qualified exact-layout BF16 GEMM preset"
        atomic_json(runtime, manifest)
        record = dict(
            status="qualified bounded",
            source=source,
            library_sha256=sha(library),
            csv_sha256=sha(new_csv),
            installed_gates=report["installed_gates"],
            recovery=str(RECOVERY),
            measured_model_ratios=json.loads(
                (WS / "logs/gemm-n32-model-candidate-r1/result.json").read_text()
            )["paired_wall_ratios"],
        )
        atomic_json(WS / "SOURCE_GEMM_N32.json", record)
        atomic_json(RELEASE / "SOURCE_GEMM_N32.json", record)
        protected = {
            p: sha(Path(p)) if p in allowed else d for p, d in old_protected.items()
        }
        for p in [
            launcher,
            helper,
            new_csv,
            WS / "SOURCE_GEMM_N32.json",
            RELEASE / "SOURCE_GEMM_N32.json",
            library,
        ]:
            protected[str(p)] = sha(p)
        for relative, digest in identities.items():
            protected[str(PREFIX / relative)] = digest
        report.update(
            status="passed",
            protected_sha256=protected,
            runtime_sha256=sha(runtime),
            links=baseline["links"],
            aot_caches=baseline["aot_caches"],
            torch_target=baseline["torch_target"],
            rocm_target=baseline["rocm_target"],
            intentional_changed_sha256={
                p: dict(before=d, after=sha(Path(p)))
                for p, d in old_protected.items()
                if p in allowed
            },
            kernel_log_unchanged_entire_batch=True,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            kernel=os.uname().release,
            disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
            running_agent_jobs=[],
            prefix_changed=True,
        )
    except Exception as error:
        report.update(status="failed-rolled-back", error=repr(error))
        if changed:
            replace(library, (RECOVERY / "librocblas.installed.so.5.8").read_bytes())
            replace(runtime, old_runtime)
            replace(launcher, (RECOVERY / "bc250-torch").read_bytes())
            replace(helper, (RECOVERY / "bc250_torch.helper.py").read_bytes())
            if new_csv.exists():
                os.replace(new_csv, RELEASE / "rocblas-gemm-n32.rejected.csv")
        raise
    finally:
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"


if __name__ == "__main__":
    main()
