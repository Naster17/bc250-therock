#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify the relocated MIOpen/CK package with build paths absent."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.artifact.resolve()
    out = args.output.parent
    entry = root / "bin/bc250-ck.py"
    result = dict(status="running", checks=[], probes={})

    def execute(label, command, env=None, success=True):
        proc = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=240
        )
        (out / (label + "-stdout.log")).write_text(proc.stdout)
        (out / (label + "-stderr.log")).write_text(proc.stderr)
        if success:
            proc.check_returncode()
        else:
            assert proc.returncode == 2 and "bc250-ck:" in proc.stderr, (
                label,
                proc.returncode,
                proc.stderr[-500:],
            )
        return proc

    def dbhashes(db):
        return {p.name: sha(p) for p in db.glob("*.txt")}

    try:
        ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
        for name in (
            "source",
            "build",
            "tmp/ck-plugin-b0710/build-r2",
            "tmp/ck-plugin-b0711/native-fix-r2",
            "tmp/ck-plugin-b0711/build-r3",
        ):
            assert not (ws / name).exists(), name
        result["checks"].append(
            "extracted-only-build-source-and-candidate-paths-absent"
        )
        proc = execute("identity", [sys.executable, str(entry), "check"])
        identity = json.loads(proc.stdout.strip().splitlines()[-1])
        assert identity["status"] == "passed"
        assert identity["runtime_libraries"]["libMIOpen.so.1"]["path"] == str(
            root / "runtime/libMIOpen.so.1"
        )
        result["identity"] = identity
        result["checks"].append(
            "actual-mapped-private-MIOpen-native-ABI-payload-and-API5-identity"
        )

        # The controller precreates same-filesystem immutable payload fixtures.
        for kind in ("native", "architecture", "payload", "escape", "extra", "symlink"):
            bad = Path(os.environ["TMPDIR"]) / "identity-fixtures" / ("bad-" + kind)
            assert bad.is_dir()
            manifest = bad / "manifest.json"
            manifest.chmod(0o644)
            data = json.loads(manifest.read_text())
            if kind == "native":
                data["runtime_libraries"]["libamdhip64.so.7"]["sha256"] = "0" * 64
            elif kind == "architecture":
                data["architecture"] = "gfx1100"
            elif kind == "payload":
                data["files"]["lib/libMIOpenCKGroupedConv_gfx1013.so"] = "0" * 64
            elif kind == "escape":
                data["files"]["../outside"] = "0" * 64
            elif kind == "extra":
                (bad / "unexpected.txt").write_text("unexpected\n")
            else:
                (bad / "unexpected-link").symlink_to(root / "README.md")
            manifest.write_text(json.dumps(data, indent=2) + "\n")
            (out / ("refusal-" + kind + "-manifest.json")).write_bytes(
                manifest.read_bytes()
            )
            proc = execute(
                "refusal-" + kind,
                [sys.executable, str(bad / "bin/bc250-ck.py"), "check"],
                success=False,
            )
            expected = dict(
                native="runtime identity mismatch",
                architecture="identity/architecture",
                payload="hash mismatch",
                escape="escaping payload",
                extra="unexpected/missing",
                symlink="unexpected/missing",
            )[kind]
            assert expected in proc.stderr, (kind, proc.stderr)
        result["checks"].append(
            "native-architecture-payload-escape-extra-and-symlink-refusals"
        )
        cohort = os.environ["BC250_CK_TEST_COHORT"]
        db = out / "seeded-db"
        execute(
            "seed",
            [
                sys.executable,
                str(entry),
                "seed-db",
                "--cohort",
                cohort,
                "--destination",
                str(db),
            ],
        )
        snapshot = dbhashes(db)
        assert len(snapshot) == 2
        for file in db.glob("*.txt"):
            lines = file.read_text().splitlines()
            assert (
                len(lines) == 8
                and sum("FP16" in x for x in lines) == 6
                and sum("FP32" in x for x in lines) == 2
            )
        execute(
            "seed-refusal",
            [
                sys.executable,
                str(entry),
                "seed-db",
                "--cohort",
                cohort,
                "--destination",
                str(db),
            ],
            success=False,
        )
        assert dbhashes(db) == snapshot
        result["checks"].append(
            "genuine-eight-key-mixed-dtype-seed-and-existing-destination-refusal"
        )
        env = dict(
            os.environ,
            MIOPEN_LOG_LEVEL="6",
            MIOPEN_FIND_ENFORCE="NONE",
            MIOPEN_USER_DB_PATH=str(db),
            MIOPEN_CUSTOM_CACHE_DIR=str(out / "kernel-cache"),
        )
        reference = out / "references.json"
        execute(
            "references",
            [
                sys.executable,
                str(entry),
                "run",
                str(root / "validation/references.py"),
                "--output",
                str(reference),
            ],
            dict(env, MIOPEN_USER_DB_PATH=str(out / "reference-db")),
        )
        refs = json.loads(reference.read_text())
        assert refs["status"] == "passed" and len(refs["checks"]) == 12
        result["references"] = refs
        result["checks"].append(
            "entry-same-process-FP32-FP16-nine-IDs-mixed-handle-contracts-and-gradients-12"
        )
        # Reuse the artifact's environment construction for independent public probes.
        spec = importlib.util.spec_from_file_location("entry", entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        probe_env = dict(
            module.environment(root),
            **{k: v for k, v in env.items() if k.startswith("MIOPEN_")}
        )
        for kind, count in [("b076", 2), ("b0711", 6)]:
            proc = execute(
                "replay-" + kind,
                [str(root / "probes" / ("probe-" + kind)), "--diagnostic"],
                probe_env,
            )
            value = json.loads(proc.stdout.strip().splitlines()[-1])
            assert (
                value["status"] == "passed"
                and len(value["rows"]) == count
                and value["guards"]
            )
            assert (
                "Search Started:" not in proc.stderr
                and proc.stderr.count(
                    "Perf Db: record loaded: ConvHipImplicitGemmGroupFwdXdlops"
                )
                >= count
            )
            result["probes"][kind] = value
            result["checks"].append(
                "fresh-public-" + kind + "-CPU-reference-and-genuine-PerfDb-replay"
            )
        assert dbhashes(db) == snapshot, "replay changed seeded tuning records"
        probe_env.pop("MIOPEN_CK_LIB_PATH")
        proc = execute(
            "absent-FP32", [str(root / "probes/probe-b072"), "--negative"], probe_env
        )
        value = json.loads(proc.stdout.strip().splitlines()[-1])
        assert value["executions"] == 0 and value["refusals"] == 32
        result["checks"].append(
            "fresh-process-without-explicit-CK-path-refuses-FP32-private-solver"
        )
        appdir = out / "owned caller"
        appdir.mkdir()
        (appdir / "local_helper.py").write_text("VALUE=412\n")
        receipt = appdir / "receipt.json"
        app = appdir / "app.py"
        app.write_text(
            'import os,sys,json,torch,local_helper\nfrom pathlib import Path\nassert local_helper.VALUE==412 and sys.argv[1:]==["literal argument", "--owned"]\nassert os.environ["BC250_CK_CALLER"]=="preserved"\nx=torch.arange(35,device="cuda",dtype=torch.float32).requires_grad_();x.square().sum().backward();torch.cuda.synchronize()\ntorch.testing.assert_close(x.grad.cpu(),2*torch.arange(35,dtype=torch.float32),rtol=0,atol=0)\nPath('
            + repr(str(receipt))
            + ').write_text(json.dumps(dict(status="passed",plugin=os.environ["MIOPEN_CK_LIB_PATH"],cwd=os.getcwd())))\n'
        )
        execute(
            "caller",
            [
                sys.executable,
                str(entry),
                "run",
                str(app),
                "literal argument",
                "--owned",
            ],
            dict(
                os.environ,
                BC250_CK_CALLER="preserved",
                MIOPEN_CK_LIB_PATH="/intentionally-invalid",
            ),
        )
        value = json.loads(receipt.read_text())
        assert value == dict(status="passed", plugin=str(root / "lib"), cwd=os.getcwd())
        result["checks"].append(
            "worker-owned-script-argv-importdir-env-cwd-native-backward-and-extracted-plugin"
        )
        assert len(result["checks"]) == 9
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
