#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-04 workstation orchestrator: bounded attention qualification/benchmark.

Stages the attention package + the qualified guardian envelope onto .104, gates
on stock clocks (pp_dpm_sclk ceiling <=2000 MHz) and an idle board, then runs one
GPU part under the PLAN2-01/02/03 envelope (SH lock, 600 s nonce-acked boot-guard,
460 s child, timeout 430 backstop, kernel/registry identity, foreign-container
refusal). Collects the report + guardian evidence locally. Single host; owned
synthetic fixtures only; no user application; no default/release/source change.
Root via ssh -tt doas, PIN recovered at runtime (never persisted).

Usage:
    python3 run_attention.py --label attn-r1 --script qualify --part 1
    python3 run_attention.py --label attn-r2 --script qualify --part 2
    python3 run_attention.py --label attn-r3 --script bench --part 1
    python3 run_attention.py --label attn-r4 --script bench --part 2
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
TASK = "/home/nik/bc250work/attention-plan2-04"
EVIDENCE = Path(
    "/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/attention-plan2-04"
)
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/attention-plan2-04")
SESSION_TIMEOUT = base.SESSION_TIMEOUT
# Owned lora-smoke env (PLAN2-01 soak pattern): transformers for the integration
# check only; it carries no torch/triton, so the release torch stays authoritative.
ENV_SITE = "/home/nik/bc250-projects/lora-smoke/env/lib/python3.12/site-packages"
PACKAGE_FILES = [
    "__init__.py",
    "fixtures.py",
    "qualify_attention.py",
    "bench_attention.py",
]


def report_path(script: str, part: int) -> str:
    return f"{TASK}/outputs/{script}-part{part}.json"


def stage() -> None:
    base.ssh(
        f"mkdir -p {TASK}/tools/attention_plan2_04 {TASK}/configs "
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
        f"{TASK}/tools/attention_plan2_04/",
    )
    print("ATTN_STAGE package + guardian staged", flush=True)


def guardian_config(label: str, script: str, part: int, diagnose: str = "") -> dict:
    argv = [
        "--env",
        f"PYTHONPATH={ENV_SITE}",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "OMP_NUM_THREADS=4",
        "--shm-size",
        "256m",
        f"{TASK}/tools/attention_plan2_04/{script}_attention.py",
        "--output",
        report_path(script, part),
        "--part",
        str(part),
    ]
    if diagnose:
        argv.append(f"--{diagnose}")
    config = {
        "mode": "single",
        "label": label,
        "argv": argv,
        "rank": 0,
        "evidence": f"{TASK}/evidence/{label}",
        "cwd": TASK,
    }
    if not diagnose:
        config["audit"] = report_path(script, part)
    return config


def run_guardian(
    label: str, script: str, part: int, pin: str, diagnose: str = ""
) -> bool:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    config = guardian_config(label, script, part, diagnose)
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
    print(f"ATTN_GUARDIAN ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def collect(label: str, script: str, part: int, preflight: dict) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    final = json.loads(base.ssh(f"cat {TASK}/evidence/{label}/final.json"))
    try:
        report = json.loads(base.ssh(f"cat {report_path(script, part)}"))
    except RuntimeError:
        report = None
    (EVIDENCE / f"guardian-{label}.json").write_text(json.dumps(final, indent=2) + "\n")
    if report is not None:
        (EVIDENCE / f"{script}-{label}.json").write_text(
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
        f"ATTN_COLLECT label={label} guardian={final.get('status')} "
        f"checks={summary.get('passed')}/{summary.get('total')} "
        f"all_passed={summary.get('all_passed')}",
        flush=True,
    )
    return {"guardian": final, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="attn-r1")
    parser.add_argument("--script", choices=("qualify", "bench"), required=True)
    parser.add_argument("--part", type=int, choices=(1, 2), required=True)
    parser.add_argument("--diagnose", choices=("diagnose", "diagnose2"), default="")
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
    ok = run_guardian(label, args.script, args.part, pin, args.diagnose)
    result = collect(label, args.script, args.part, preflight)
    if args.diagnose:
        passed = ok and result["guardian"].get("status") == "passed"
    else:
        passed = (
            ok
            and result["guardian"].get("status") == "passed"
            and (result["report"] or {}).get("summary", {}).get("all_passed")
        )
    print(f"ATTN_RESULT passed={bool(passed)}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
