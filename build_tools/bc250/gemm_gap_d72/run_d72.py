#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D72 workstation orchestrator: GEMM in-model-vs-fixture gap diagnostic.

Phases: idle check -> stage tools + the archived D71 census (owned evidence) ->
three guarded fixture microbenchmarks (alt-impl flags A/B, sustained-load
throttle probe, cold/hot locality probe) -> collect -> summarize attribution.
Single host (.104); measurement only — NO model training, NO full-model profile
(avoids the D71 r6 host-thrash vector), NO source/default/release/preset change,
NO user data. Reuses the D71-hardened envelope (timeout 430 podman backstop,
460 s guardian, 900 s session).

Root steps use ssh -tt doas with the PIN recovered at runtime from the Codex
rollout log (never printed, never persisted). Recreate this driver per
milestone; delete it at closeout. Only additive task paths are used.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pexpect

HOST = "192.168.1.104"
TASK = "/home/nik/bc250work/gemm-gap-d72"
PROJECT = "/home/nik/bc250-projects/lora-smoke/project"
ENV_SITE = "/home/nik/bc250-projects/lora-smoke/env/lib/python3.12/site-packages"
LOCAL_TOOLS = Path(__file__).resolve().parent
BC250 = LOCAL_TOOLS.parent
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/gemm-gap-d72")
EVIDENCE = Path("/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/gemm-gap-d72")
CENSUS_SRC = Path(
    "/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/gemm-efficiency-d71/census-1024.json"
)
PROFILE_SRC = Path(
    "/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/gemm-efficiency-d71/profile-shapes-1024.json"
)
SESSION_TIMEOUT = 900
MODES = ("alt-impl", "sustained", "locality")

sys.path.insert(0, str(LOCAL_TOOLS))
import summarize_d72 as summarize  # noqa: E402


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
        ["ssh", f"nik@{HOST}", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh failed: {proc.stderr[-800:]}")
    return proc.stdout


def scp_to(sources: list[str], dest: str) -> None:
    subprocess.run(
        ["scp", "-q", *sources, f"nik@{HOST}:{dest}"], check=True, timeout=300
    )


def base_env() -> list[str]:
    return [
        "--env",
        f"PYTHONPATH={ENV_SITE}",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TOKENIZERS_PARALLELISM=false",
        "--env",
        "OMP_NUM_THREADS=4",
    ]


def guardian_config(name: str) -> dict:
    output = f"{TASK}/evidence/d72-{name}/{name}.json"
    argv = [
        *base_env(),
        "--shm-size",
        "256m",
        f"{TASK}/tools/probe_gap_d72.py",
        "--mode",
        name,
        "--census",
        f"{TASK}/configs/census-1024.json",
        "--profile",
        f"{TASK}/configs/profile-shapes-1024.json",
        "--output",
        output,
    ]
    return {
        "mode": "single",
        "label": f"d72-{name}",
        "argv": argv,
        "rank": 0,
        "evidence": f"{TASK}/evidence/d72-{name}",
        "cwd": PROJECT,
    }


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


def phase_passed(name: str) -> bool:
    try:
        final = json.loads(ssh(f"cat {TASK}/evidence/d72-{name}/final.json"))
    except RuntimeError:
        return False
    return final.get("status") == "passed"


def clean_failed(name: str, pin: str) -> None:
    child = spawn_root(f"doas rm -rf {TASK}/evidence/d72-{name}", pin, timeout=60)
    try:
        child.expect(pexpect.EOF, timeout=60)
    finally:
        child.close()
    if child.exitstatus != 0:
        raise RuntimeError(f"cleanup of {name} evidence failed")


def run_guardian(name: str, pin: str) -> bool:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    config = guardian_config(name)
    config_path = CONFIG_SCRATCH / f"guard-{name}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    scp_to([str(config_path)], f"{TASK}/configs/")
    command = f"doas python3 {TASK}/tools/guard_d72.py {TASK}/configs/guard-{name}.json"
    child = spawn_root(command, pin)
    log = (CONFIG_SCRATCH / f"guard-{name}.log").open("w")
    child.logfile_read = log
    try:
        child.expect(pexpect.EOF, timeout=SESSION_TIMEOUT)
    finally:
        child.close()
        log.close()
    print(f"D72_CASE {name} ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def run_phase(name: str, pin: str, statuses: dict) -> None:
    required = f"{TASK}/evidence/d72-{name}/{name}.json"
    passed = phase_passed(name)
    if passed:
        passed = bool(ssh(f"test -f {required} && echo yes || true").strip())
    if passed:
        print(f"D72_CASE {name} SKIP (already passed)", flush=True)
        statuses[name] = True
        return
    if ssh(f"test -d {TASK}/evidence/d72-{name} && echo yes || true").strip():
        clean_failed(name, pin)
    statuses[name] = run_guardian(name, pin)


def idle_check() -> None:
    foreign = ssh("podman ps -q").split()
    if foreign:
        raise RuntimeError(f"board not idle; foreign containers: {foreign}")
    boot = ssh("cat /proc/sys/kernel/random/boot_id").strip()
    print(f"D72_IDLE boot_id={boot} containers=0", flush=True)


def stage() -> None:
    ssh(f"mkdir -p {TASK}/tools {TASK}/configs {TASK}/evidence")
    scp_to(
        [
            str(LOCAL_TOOLS / "guard_d72.py"),
            str(LOCAL_TOOLS / "bounded_launch.py"),
            str(BC250 / "probe_gap_d72.py"),
            str(BC250 / "bench_gemm_d71.py"),
        ],
        f"{TASK}/tools/",
    )
    scp_to([str(CENSUS_SRC)], f"{TASK}/configs/census-1024.json")
    scp_to([str(PROFILE_SRC)], f"{TASK}/configs/profile-shapes-1024.json")
    print("D72_STAGE tools + D71 census/profile staged", flush=True)


def collect(statuses: dict) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in MODES:
        remote = f"{TASK}/evidence/d72-{name}/{name}.json"
        report = json.loads(ssh(f"cat {remote}"))
        reports[name] = report
        (EVIDENCE / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        final = json.loads(ssh(f"cat {TASK}/evidence/d72-{name}/final.json"))
        (EVIDENCE / f"{name}-guardian.json").write_text(
            json.dumps(final, indent=2) + "\n"
        )
    summary = summarize.build_summary(
        reports["alt-impl"], reports["sustained"], reports["locality"]
    )
    summary["guardian_status"] = statuses
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {}
    for path in sorted(EVIDENCE.glob("*.json")):
        if path.name == "artifacts.json":
            continue
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (EVIDENCE / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> int:
    pin = recover_pin()
    idle_check()
    stage()
    statuses: dict = {}
    for name in MODES:
        run_phase(name, pin, statuses)
        if not statuses[name]:
            raise RuntimeError(f"{name} failed; stopping")
    summary = collect(statuses)
    print("D72_RATIOS " + json.dumps(summary["ratios"]), flush=True)
    print(
        f"D72_ATTRIBUTION {summary['attribution']} primary={summary['primary_cause']}",
        flush=True,
    )
    for finding in summary["findings"]:
        print(f"D72_FINDING {finding}", flush=True)
    print("D72_STATUS " + summary["status"], flush=True)
    print("D72_DONE", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
