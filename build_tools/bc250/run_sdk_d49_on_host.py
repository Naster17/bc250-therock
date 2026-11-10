#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the reversible D49 batch, restoring the original selection on failure."""
import subprocess

from stage_sdk_d49_on_host import WORK, ROOT, sha, write

TOOLS = WORK / "tools/sdk-d49"
EVIDENCE = WORK / "evidence/sdk-d49-controller"


def main():
    EVIDENCE.mkdir(exist_ok=False)
    log = dict(
        status="running", steps=[], sources={p.name: sha(p) for p in TOOLS.glob("*.py")}
    )
    before = WORK / "evidence/sdk-d49-install/activate-r1/before.json"

    def step(script, *args):
        cmd = ["python3", str(TOOLS / script), *map(str, args)]
        with (EVIDENCE / (str(len(log["steps"])) + "-step.log")).open("w") as stream:
            result = subprocess.run(
                cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=540
            )
        log["steps"].append(dict(command=cmd, exit=result.returncode))
        write(EVIDENCE / "result.json", log)
        assert result.returncode == 0, cmd

    try:
        step("stage_sdk_d49_on_host.py")
        # Identity checks use real CK/MIGraphX/ORT registered() implementations,
        # with no GPU workload or package/registration identity relaxation.
        code = "import sys;sys.path.insert(0,sys.argv[1]);import bc250_ck,bc250_migraphx,bc250_ort;from bc250_torch import LaunchOptions;[(m.registered(LaunchOptions(runtime_profile=p))) for p in ('daily','rccl-single') for m in (bc250_ck,bc250_migraphx,bc250_ort)];print('six provider registrations passed')"
        subprocess.run(
            ["python3", "-c", code, str(ROOT / "tools")], check=True, timeout=60
        )
        step(
            "activate_sdk_d49_on_host.py",
            "--action",
            "activate",
            "--label",
            "activate-r1",
        )
        step("guard_sdk_d49_on_host.py", "--phase", "activation-r1")
        step(
            "activate_sdk_d49_on_host.py",
            "--action",
            "rollback",
            "--label",
            "rollback-r1",
            "--from-transaction",
            before,
        )
        step("guard_sdk_d49_on_host.py", "--phase", "rollback-r1")
        step(
            "activate_sdk_d49_on_host.py",
            "--action",
            "activate",
            "--label",
            "activate-r2",
        )
        step("guard_sdk_d49_on_host.py", "--phase", "final-r1")
        log["status"] = "passed"
    except BaseException as error:
        log.update(status="failed", error=repr(error))
        if before.exists():
            try:
                step(
                    "activate_sdk_d49_on_host.py",
                    "--action",
                    "rollback",
                    "--label",
                    "failure-rollback",
                    "--from-transaction",
                    before,
                )
                log["selection_restored"] = True
            except BaseException as restore_error:
                log["restore_error"] = repr(restore_error)
        raise
    finally:
        write(EVIDENCE / "result.json", log)


if __name__ == "__main__":
    main()
