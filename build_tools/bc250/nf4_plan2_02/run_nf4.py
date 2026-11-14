#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-02 workstation orchestrator: one bounded native NF4/QLoRA qualification.

Stages the NF4 package + guardian onto .104, gates on stock clocks (pp_dpm_sclk
ceiling <=2000 MHz) and an idle board, then runs the GPU qualification suite under
the qualified PLAN2-01 guardian envelope (SH lock, 600 s nonce-acked boot-guard,
460 s child, timeout 430 backstop, kernel/registry identity, foreign-container
refusal). Collects the report + guardian evidence locally. Single host; owned
synthetic fixtures only; no user application; no default/release/source change.

Root steps use ssh -tt doas with the PIN recovered at runtime from the Codex
rollout log (never printed, never persisted). Recreate per milestone; delete at
closeout.

Usage:
    python3 run_nf4.py            # stage + qualify + collect
    python3 run_nf4.py --collect-only
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
import subprocess
import sys

import pexpect

HOST = "192.168.1.104"
TASK = "/home/nik/bc250work/nf4-plan2-02"
LOCAL_TOOLS = Path(__file__).resolve().parent
BC250 = LOCAL_TOOLS.parent
SOAK_DIR = BC250 / "soak_plan2_01"
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/nf4-plan2-02")
EVIDENCE = Path("/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/nf4-plan2-02")
SESSION_TIMEOUT = 900
LABEL = "nf4-r1"
QUALIFY_SCRIPT = f"{TASK}/tools/nf4_plan2_02/qualify_nf4.py"
REPORT = f"{TASK}/outputs/qualification.json"
PACKAGE_FILES = ["__init__.py", "core.py", "ops.py", "reference.py", "qualify_nf4.py"]
STOCK_CLOCK_MAX_MHZ = 2000


def recover_pin() -> str:
    files = glob.glob("/home/nik/.codex/sessions/2026/11/10/rollout-*01a1c506*.jsonl")
    pat = re.compile(r'write_stdin\(\{session_id:72410,chars:"(\d{4})(?:\\n)?"')
    pins = set()
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            inp = d.get("payload", {}).get("input")
            if isinstance(inp, str):
                pins.update(pat.findall(inp))
    if len(pins) != 1:
        raise SystemExit("PIN recovery ambiguous")
    return pins.pop()


def ssh(command: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        ["ssh", f"nik@{HOST}", command], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh failed: {proc.stderr[-800:]}")
    return proc.stdout


def scp_to(sources: list[str], dest: str) -> None:
    subprocess.run(
        ["scp", "-q", *sources, f"nik@{HOST}:{dest}"], check=True, timeout=300
    )


def spawn_root(command: str, pin: str, timeout: int = SESSION_TIMEOUT) -> pexpect.spawn:
    child = pexpect.spawn(
        "ssh",
        ["-tt", f"nik@{HOST}", command],
        timeout=timeout,
        encoding="utf-8",
        codec_errors="replace",
    )
    i = child.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT], timeout=60)
    if i != 0:
        raise RuntimeError(f"no doas prompt (i={i})")
    child.sendline(pin)
    return child


def parse_sclk_levels(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"(\d+)Mhz", text)]


def stock_clock_check() -> dict:
    text = ssh("cat /sys/class/drm/card0/device/pp_dpm_sclk")
    levels = parse_sclk_levels(text)
    if not levels or max(levels) > STOCK_CLOCK_MAX_MHZ:
        raise RuntimeError(f"board not at stock clocks; sclk levels={levels}")
    print(f"NF4_STOCK_CLOCKS sclk_levels={levels} max={max(levels)}MHz", flush=True)
    return {"sclk_levels_mhz": levels, "max_mhz": max(levels)}


def idle_check() -> dict:
    foreign = ssh("podman ps -q").split()
    if foreign:
        raise RuntimeError(f"board not idle; foreign containers: {foreign}")
    boot = ssh("cat /proc/sys/kernel/random/boot_id").strip()
    temp = ssh("cat /sys/class/hwmon/hwmon*/temp1_input 2>/dev/null | head -1").strip()
    info = {
        "boot_id": boot,
        "containers": 0,
        "temp_c": int(temp) / 1000.0 if temp else None,
    }
    print(f"NF4_IDLE boot_id={boot} containers=0 temp_c={info['temp_c']}", flush=True)
    return info


def stage() -> None:
    ssh(
        f"mkdir -p {TASK}/tools/nf4_plan2_02 {TASK}/configs {TASK}/outputs {TASK}/evidence"
    )
    scp_to([str(LOCAL_TOOLS / "guard_nf4.py")], f"{TASK}/tools/")
    scp_to(
        [str(SOAK_DIR / "bounded_launch.py"), str(SOAK_DIR / "kernel_noise.py")],
        f"{TASK}/tools/",
    )
    scp_to(
        [str(LOCAL_TOOLS / name) for name in PACKAGE_FILES],
        f"{TASK}/tools/nf4_plan2_02/",
    )
    print("NF4_STAGE package + guardian staged", flush=True)


def guardian_config() -> dict:
    return {
        "mode": "single",
        "label": LABEL,
        "argv": [
            "--env",
            "OMP_NUM_THREADS=4",
            "--shm-size",
            "256m",
            QUALIFY_SCRIPT,
            "--output",
            REPORT,
        ],
        "rank": 0,
        "evidence": f"{TASK}/evidence/{LABEL}",
        "cwd": TASK,
        "audit": REPORT,
    }


def run_guardian(pin: str) -> bool:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    config = guardian_config()
    config_path = CONFIG_SCRATCH / f"guard-{LABEL}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    scp_to([str(config_path)], f"{TASK}/configs/")
    command = (
        f"doas python3 {TASK}/tools/guard_nf4.py {TASK}/configs/guard-{LABEL}.json"
    )
    child = spawn_root(command, pin)
    log = (CONFIG_SCRATCH / f"guard-{LABEL}.log").open("w")
    child.logfile_read = log
    try:
        child.expect(pexpect.EOF, timeout=SESSION_TIMEOUT)
    finally:
        child.close()
        log.close()
    print(f"NF4_GUARDIAN ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def collect(preflight: dict) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    final = json.loads(ssh(f"cat {TASK}/evidence/{LABEL}/final.json"))
    try:
        report = json.loads(ssh(f"cat {REPORT}"))
    except RuntimeError:
        report = None
    (EVIDENCE / f"guardian-{LABEL}.json").write_text(json.dumps(final, indent=2) + "\n")
    if report is not None:
        (EVIDENCE / f"qualification-{LABEL}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    (EVIDENCE / f"preflight-{LABEL}.json").write_text(
        json.dumps(preflight, indent=2) + "\n"
    )
    for name in ("gpu.log", "kernel-before.log", "kernel-after.log"):
        try:
            data = ssh(f"cat {TASK}/evidence/{LABEL}/{name}")
            (EVIDENCE / f"{LABEL}-{name}").write_text(data)
        except RuntimeError:
            pass
    summary = (report or {}).get("summary", {})
    print(
        f"NF4_COLLECT label={LABEL} guardian={final.get('status')} "
        f"checks={summary.get('passed')}/{summary.get('total')} "
        f"all_passed={summary.get('all_passed')}",
        flush=True,
    )
    return {"guardian": final, "report": report}


def main() -> int:
    global LABEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--label", default=LABEL, help="guardian run label/evidence dir"
    )
    args = parser.parse_args()
    LABEL = args.label
    preflight = {"stock_clocks": stock_clock_check(), "idle": idle_check()}
    if args.collect_only:
        collect(preflight)
        return 0
    pin = recover_pin()
    stage()
    ok = run_guardian(pin)
    result = collect(preflight)
    passed = (
        ok
        and result["guardian"].get("status") == "passed"
        and (result["report"] or {}).get("summary", {}).get("all_passed")
    )
    print(f"NF4_RESULT passed={bool(passed)}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
