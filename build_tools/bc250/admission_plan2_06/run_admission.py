#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-06 workstation orchestrator: bounded broader-model admission.

Stages the admission package + the D62 helper module + the qualified guardian
envelope onto .104, gates on stock clocks and an idle board, runs one
model/dtype admission under the PLAN2-01..05 envelope, collects report +
guardian evidence. Owned synthetic fixtures only; no app/source/default/release
change. Root via ssh -tt doas, PIN recovered at runtime (never persisted).

Usage:
    python3 run_admission.py --label ad-r1 --model unet2d-tiny --dtype float32
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
TASK = "/home/nik/bc250work/admission-plan2-06"
EVIDENCE = Path(
    "/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/admission-plan2-06"
)
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/admission-plan2-06")
SESSION_TIMEOUT = base.SESSION_TIMEOUT
ENV_SITE = "/home/nik/bc250-projects/lora-smoke/env/lib/python3.12/site-packages"
PACKAGE_FILES = ["__init__.py", "models.py", "admit.py"]


def report_path(model: str, dtype: str) -> str:
    return f"{TASK}/outputs/admit-{model}-{dtype}.json"


def stage() -> None:
    base.ssh(
        f"mkdir -p {TASK}/tools/admission_plan2_06 {TASK}/configs "
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
    # admit.py imports admit_model_d62 from its parent dir (TASK/tools).
    base.scp_to([str(BC250 / "admit_model_d62.py")], f"{TASK}/tools/")
    base.scp_to(
        [str(LOCAL_TOOLS / name) for name in PACKAGE_FILES],
        f"{TASK}/tools/admission_plan2_06/",
    )
    print("AD_STAGE package + D62 helper + guardian staged", flush=True)


def guardian_config(label: str, model: str, dtype: str) -> dict:
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
            f"{TASK}/tools/admission_plan2_06/admit.py",
            "--model",
            model,
            "--dtype",
            dtype,
            "--output",
            report_path(model, dtype),
        ],
        "rank": 0,
        "evidence": f"{TASK}/evidence/{label}",
        "cwd": TASK,
        "audit": report_path(model, dtype),
    }


def run_guardian(label: str, model: str, dtype: str, pin: str) -> bool:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    config = guardian_config(label, model, dtype)
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
    print(f"AD_GUARDIAN ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def collect(label: str, model: str, dtype: str, preflight: dict) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    final = json.loads(base.ssh(f"cat {TASK}/evidence/{label}/final.json"))
    try:
        report = json.loads(base.ssh(f"cat {report_path(model, dtype)}"))
    except RuntimeError:
        report = None
    (EVIDENCE / f"guardian-{label}.json").write_text(json.dumps(final, indent=2) + "\n")
    if report is not None:
        (EVIDENCE / f"admit-{label}.json").write_text(
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
        f"AD_COLLECT label={label} guardian={final.get('status')} "
        f"model={model}/{dtype} status={(report or {}).get('status')} "
        f"checks={summary.get('passed')}/{summary.get('total')}",
        flush=True,
    )
    return {"guardian": final, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="ad-r1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
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
    ok = run_guardian(label, args.model, args.dtype, pin)
    result = collect(label, args.model, args.dtype, preflight)
    passed = (
        ok
        and result["guardian"].get("status") == "passed"
        and (result["report"] or {}).get("summary", {}).get("all_passed")
    )
    print(f"AD_RESULT passed={bool(passed)}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
