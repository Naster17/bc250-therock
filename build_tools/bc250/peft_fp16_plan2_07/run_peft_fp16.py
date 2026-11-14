#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-07 workstation orchestrator: bounded FP16 PEFT qualification.

Stages the package + qualified guardian envelope onto .104, gates on stock
clocks and an idle board, runs the FP16 PEFT battery under the PLAN2-01..06
envelope, collects report + guardian evidence. Owned synthetic fixture only; no
app/source/default/release change; frozen FSDP helpers untouched. Root via
ssh -tt doas, PIN recovered at runtime (never persisted).

Usage:
    python3 run_peft_fp16.py --label fp-r1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BC250 = Path(__file__).resolve().parents[1]
NF4_DIR = BC250 / "nf4_plan2_02"
sys.path.insert(0, str(NF4_DIR))

import run_nf4 as base  # noqa: E402

LOCAL_TOOLS = Path(__file__).resolve().parent
TASK = "/home/nik/bc250work/peft-fp16-plan2-07"
EVIDENCE = Path(
    "/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/peft-fp16-plan2-07"
)
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/peft-fp16-plan2-07")
SESSION_TIMEOUT = base.SESSION_TIMEOUT
ENV_SITE = "/home/nik/bc250-projects/lora-smoke/env/lib/python3.12/site-packages"
PACKAGE_FILES = ["__init__.py", "policy.py", "qualify_peft_fp16.py"]
REPORT = f"{TASK}/outputs/qualify.json"


def stage() -> None:
    base.ssh(
        f"mkdir -p {TASK}/tools/peft_fp16_plan2_07 {TASK}/configs "
        f"{TASK}/outputs {TASK}/evidence"
    )
    base.scp_to([str(NF4_DIR / "guard_nf4.py")], f"{TASK}/tools/")
    base.scp_to(
        [
            str(base.SOAK_DIR / "bounded_launch.py"),
            str(base.SOAK_DIR / "kernel_noise.py"),
        ],
        f"{TASK}/tools/",
    )
    base.scp_to(
        [str(LOCAL_TOOLS / name) for name in PACKAGE_FILES],
        f"{TASK}/tools/peft_fp16_plan2_07/",
    )
    print("FP16PEFT_STAGE package + guardian staged", flush=True)


def guardian_config(label: str) -> dict:
    return {
        "mode": "single",
        "label": label,
        "argv": [
            "--env",
            f"PYTHONPATH={ENV_SITE}",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "OMP_NUM_THREADS=4",
            "--shm-size",
            "256m",
            f"{TASK}/tools/peft_fp16_plan2_07/qualify_peft_fp16.py",
            "--output",
            REPORT,
        ],
        "rank": 0,
        "evidence": f"{TASK}/evidence/{label}",
        "cwd": TASK,
        "audit": REPORT,
    }


def run_guardian(label: str, pin: str) -> bool:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    config = guardian_config(label)
    config_path = CONFIG_SCRATCH / f"guard-{label}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    base.scp_to([str(config_path)], f"{TASK}/configs/")
    command = (
        f"doas python3 {TASK}/tools/guard_nf4.py {TASK}/configs/guard-{label}.json"
    )
    child = base.spawn_root(command, pin)
    log = (CONFIG_SCRATCH / f"guard-{label}.log").open("w")
    child.logfile_read = log
    try:
        child.expect(base.pexpect.EOF, timeout=SESSION_TIMEOUT)
    finally:
        child.close()
        log.close()
    print(f"FP16PEFT_GUARDIAN ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def collect(label: str, preflight: dict) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    final = json.loads(base.ssh(f"cat {TASK}/evidence/{label}/final.json"))
    try:
        report = json.loads(base.ssh(f"cat {REPORT}"))
    except RuntimeError:
        report = None
    (EVIDENCE / f"guardian-{label}.json").write_text(json.dumps(final, indent=2) + "\n")
    if report is not None:
        (EVIDENCE / f"qualify-{label}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    (EVIDENCE / f"preflight-{label}.json").write_text(
        json.dumps(preflight, indent=2) + "\n"
    )
    for name in ("gpu.log", "kernel-before.log", "kernel-after.log"):
        try:
            (EVIDENCE / f"{label}-{name}").write_text(
                base.ssh(f"cat {TASK}/evidence/{label}/{name}")
            )
        except RuntimeError:
            pass
    summary = (report or {}).get("summary", {})
    print(
        f"FP16PEFT_COLLECT label={label} guardian={final.get('status')} "
        f"status={(report or {}).get('status')} "
        f"checks={summary.get('passed')}/{summary.get('total')}",
        flush=True,
    )
    return {"guardian": final, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="fp-r1")
    args = parser.parse_args()
    label = args.label
    preflight = {"stock_clocks": base.stock_clock_check(), "idle": base.idle_check()}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / f"preflight-{label}.json").write_text(
        json.dumps(preflight, indent=2) + "\n"
    )
    bounds = CONFIG_SCRATCH / "BOUNDS.md"
    if bounds.exists() and not (EVIDENCE / "BOUNDS.md").exists():
        (EVIDENCE / "BOUNDS.md").write_text(bounds.read_text())
    pin = base.recover_pin()
    stage()
    ok = run_guardian(label, pin)
    result = collect(label, preflight)
    passed = (
        ok
        and result["guardian"].get("status") == "passed"
        and (result["report"] or {}).get("summary", {}).get("all_passed")
    )
    print(f"FP16PEFT_RESULT passed={bool(passed)}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
