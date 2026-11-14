#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-02 D62 admission integration: routes-only model-admission probe run.

Stages the D62 admission fixture + its sibling op probes + the NF4 package onto
.104 and runs `admit_model_d62.py --routes-only` under the same qualified guardian
envelope (audit_kind=d62 asserts status=passed and the quant-nf4 route fired).
This proves the NF4 weight-only route is admitted by the D62 framework alongside
the existing attention/conv/gated-delta/W8 routes. Owned fixtures; no model
download; no default/release/source change.

Usage:
    python3 run_d62_nf4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import run_nf4 as base

BC250 = base.BC250
LOCAL_TOOLS = base.LOCAL_TOOLS
TASK = base.TASK
EVIDENCE = base.EVIDENCE
CONFIG_SCRATCH = base.CONFIG_SCRATCH
SESSION_TIMEOUT = base.SESSION_TIMEOUT
LABEL = "nf4-d62-r1"
D62_DIR = f"{TASK}/tools/d62"
D62_SCRIPT = f"{D62_DIR}/admit_model_d62.py"
REPORT = f"{TASK}/outputs/admission-d62.json"
SIBLINGS = [
    "admit_model_d62.py",
    "qwen35_training_ops.py",
    "bc250_quant_b055.py",
    "bc250_quant_b051.py",
]


def stage() -> None:
    base.ssh(
        f"mkdir -p {D62_DIR}/nf4_plan2_02 {TASK}/configs {TASK}/outputs {TASK}/evidence"
    )
    base.scp_to([str(LOCAL_TOOLS / "guard_nf4.py")], f"{TASK}/tools/")
    base.scp_to(
        [
            str(base.SOAK_DIR / "bounded_launch.py"),
            str(base.SOAK_DIR / "kernel_noise.py"),
        ],
        f"{TASK}/tools/",
    )
    base.scp_to([str(BC250 / name) for name in SIBLINGS], f"{D62_DIR}/")
    base.scp_to(
        [str(LOCAL_TOOLS / name) for name in base.PACKAGE_FILES],
        f"{D62_DIR}/nf4_plan2_02/",
    )
    print("NF4_D62_STAGE admission fixture + NF4 package staged", flush=True)


def guardian_config() -> dict:
    return {
        "mode": "single",
        "label": LABEL,
        "argv": [
            "--env",
            "OMP_NUM_THREADS=4",
            "--shm-size",
            "256m",
            D62_SCRIPT,
            "--routes-only",
            "--output",
            REPORT,
            "--device",
            "cuda",
        ],
        "rank": 0,
        "evidence": f"{TASK}/evidence/{LABEL}",
        "cwd": TASK,
        "audit": REPORT,
        "audit_kind": "d62",
    }


def run_guardian(pin: str) -> bool:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    config = guardian_config()
    config_path = CONFIG_SCRATCH / f"guard-{LABEL}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    base.scp_to([str(config_path)], f"{TASK}/configs/")
    command = (
        f"doas python3 {TASK}/tools/guard_nf4.py {TASK}/configs/guard-{LABEL}.json"
    )
    child = base.spawn_root(command, pin)
    log = (CONFIG_SCRATCH / f"guard-{LABEL}.log").open("w")
    child.logfile_read = log
    try:
        child.expect(base.pexpect.EOF, timeout=SESSION_TIMEOUT)
    finally:
        child.close()
        log.close()
    print(f"NF4_D62_GUARDIAN ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def collect() -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    final = json.loads(base.ssh(f"cat {TASK}/evidence/{LABEL}/final.json"))
    report = json.loads(base.ssh(f"cat {REPORT}"))
    (EVIDENCE / f"guardian-{LABEL}.json").write_text(json.dumps(final, indent=2) + "\n")
    (EVIDENCE / f"admission-{LABEL}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    try:
        (EVIDENCE / f"{LABEL}-gpu.log").write_text(
            base.ssh(f"cat {TASK}/evidence/{LABEL}/gpu.log")
        )
    except RuntimeError:
        pass
    print(
        f"NF4_D62_COLLECT guardian={final.get('status')} "
        f"admission={report.get('status')} quant_nf4={report.get('routes', {}).get('quant-nf4')}",
        flush=True,
    )
    return {"guardian": final, "report": report}


def main() -> int:
    preflight = {"stock_clocks": base.stock_clock_check(), "idle": base.idle_check()}
    (EVIDENCE).mkdir(parents=True, exist_ok=True)
    (EVIDENCE / f"preflight-{LABEL}.json").write_text(
        json.dumps(preflight, indent=2) + "\n"
    )
    pin = base.recover_pin()
    stage()
    ok = run_guardian(pin)
    result = collect()
    passed = (
        ok
        and result["guardian"].get("status") == "passed"
        and result["report"].get("status") == "passed"
        and result["report"].get("routes", {}).get("quant-nf4") == "fired"
    )
    print(f"NF4_D62_RESULT passed={bool(passed)}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
