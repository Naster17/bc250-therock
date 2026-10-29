#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run extracted artifact probes without original candidates or SDK build trees."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    artifact = args.artifact
    entry = artifact / "bin/bc250-ck.py"
    r = dict(status="running", checks=[], probes={})

    def execute(label, cmd, env=None, success=True):
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
        (args.output.parent / (label + "-stdout.log")).write_text(proc.stdout)
        (args.output.parent / (label + "-stderr.log")).write_text(proc.stderr)
        if success:
            proc.check_returncode()
        else:
            assert proc.returncode == 2 and "bc250-ck:" in proc.stderr
        return proc

    try:
        ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
        assert not (ws / "tmp/ck-plugin-b073/build-r2").exists()
        assert not (ws / "tmp/ck-plugin-b076/build-r1").exists()
        assert not (ws / "source").exists()
        assert not (ws / "build").exists()
        r["checks"].append(
            "only-extracted-artifact-original-build-and-candidate-paths-absent"
        )
        execute("identity", [sys.executable, str(entry), "check"])
        r["checks"].append("native-ABI-payload-architecture-API5-66-symbol-check")
        for number, (field, value) in enumerate(
            [
                ("native", "sha"),
                ("architecture", "gfx1100"),
                ("payload", "sha"),
                ("escape", "../outside"),
            ]
        ):
            copy = args.output.parent / ("bad-" + str(number))
            shutil.copytree(artifact, copy)
            manifest = copy / "manifest.json"
            data = json.loads(manifest.read_text())
            manifest.chmod(0o644)
            if field == "native":
                data["runtime_libraries"]["libMIOpen.so.1"]["sha256"] = "0" * 64
            elif field == "architecture":
                data["architecture"] = value
            elif field == "payload":
                data["files"]["lib/libMIOpenCKGroupedConv_gfx1013.so"] = "0" * 64
            else:
                data["files"][value] = "0" * 64
            manifest.write_text(json.dumps(data, indent=2) + "\n")
            execute(
                "refusal-" + field,
                [sys.executable, str(copy / "bin/bc250-ck.py"), "check"],
                success=False,
            )
        r["checks"].append(
            "native-architecture-payload-and-escaping-path-drift-refused"
        )
        profile = (
            "rccl" if os.environ.get("BC250_CK_TEST_COHORT") == "rccl" else "daily"
        )
        db = args.output.parent / "seeded-db"
        execute(
            "seed",
            [
                sys.executable,
                str(entry),
                "seed-db",
                "--cohort",
                profile,
                "--destination",
                str(db),
            ],
        )
        before = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in db.glob("*.txt")
        }
        execute(
            "seed-refusal",
            [
                sys.executable,
                str(entry),
                "seed-db",
                "--cohort",
                profile,
                "--destination",
                str(db),
            ],
            success=False,
        )
        assert before == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in db.glob("*.txt")
        }
        r["checks"].append("genuine-example-copy-and-existing-destination-refusal")
        env = dict(
            os.environ,
            MIOPEN_CK_LIB_PATH=str(artifact / "lib"),
            MIOPEN_LOG_LEVEL="6",
            MIOPEN_FIND_ENFORCE="NONE",
            MIOPEN_USER_DB_PATH=str(db),
            MIOPEN_CUSTOM_CACHE_DIR=str(args.output.parent / "kernel-cache"),
        )
        for kind, extra in [
            ("b072", []),
            ("b073", [str(artifact / "lib/libMIOpenCKGroupedConv_gfx1013.so")]),
            ("b076", ["--diagnostic"]),
        ]:
            proc = execute(
                kind, [str(artifact / "probes" / ("probe-" + kind)), *extra], env
            )
            result = json.loads(proc.stdout.strip().splitlines()[-1])
            assert result["status"] == "passed"
            if kind == "b072":
                assert (
                    result["executions"] == 32
                    and result["refusals"] == 152
                    and result["guards"]
                )
            if kind == "b073":
                assert (
                    result["executions"] == 256
                    and result["api_refusals"] == 122
                    and result["runtime_refusals"] == 246
                    and len(result["kernel_counts"]) == 6
                    and result["guards"]
                )
            if kind == "b076":
                assert (
                    len(result["rows"]) == 2
                    and "Search Started:" not in proc.stderr
                    and "Perf Db: record loaded:" in proc.stderr
                )
            r["probes"][kind] = result
        assert before == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in db.glob("*.txt")
        }
        r["checks"].append(
            "public-solver-contracts-six-kernel-references-and-genuine-PerfDb-replay"
        )
        env.pop("MIOPEN_CK_LIB_PATH")
        proc = execute(
            "absent", [str(artifact / "probes/probe-b072"), "--negative"], env
        )
        absent = json.loads(proc.stdout.strip().splitlines()[-1])
        assert absent["executions"] == 0 and absent["refusals"] == 32
        r["checks"].append(
            "fresh-process-without-explicit-plugin-path-refuses-private-solver"
        )
        app = args.output.parent / "owned-app.py"
        receipt = args.output.parent / "owned-app-result.json"
        app.write_text(
            'import os,json,torch\nfrom pathlib import Path\nx=torch.arange(35,device="cuda",dtype=torch.float32).requires_grad_()\nx.square().sum().backward();torch.cuda.synchronize()\ntorch.testing.assert_close(x.grad.cpu(),2*torch.arange(35,dtype=torch.float32),rtol=0,atol=0)\nPath('
            + repr(str(receipt))
            + ').write_text(json.dumps(dict(status="passed",plugin=os.environ["MIOPEN_CK_LIB_PATH"])))\n'
        )
        execute(
            "run",
            [sys.executable, str(entry), "run", str(app)],
            dict(os.environ, MIOPEN_CK_LIB_PATH="/intentionally-invalid"),
        )
        assert json.loads(receipt.read_text()) == dict(
            status="passed", plugin=str(artifact / "lib")
        )
        r["checks"].append(
            "explicit-entry-runs-owned-script-with-native-backward-and-extracted-path"
        )
        assert len(r["checks"]) == 7
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
