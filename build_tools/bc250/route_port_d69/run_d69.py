#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D69 workstation orchestrator: route portability to Transformers 5.22.1.

Phases: stage tools -> verify D67 owned-data hashes (read-only reuse) ->
guarded parity fixture in the app environment -> guarded LoRA benchmark cases
(reference / installed-native / fused at 1024; reference / fused at 4096;
fused feasibility at 8192) -> collect evidence -> summarize. Single host
(.104); the two-board DDP re-check is a separate follow-up scope.

Root steps use ssh -tt doas with the PIN recovered at runtime from the Codex
rollout log (never printed, never persisted). Recreate this driver per
milestone; delete it at closeout. Only additive task paths are used
(/home/nik/bc250work/route-port-d69 and fresh output directories); the
application's original entry points, completed outputs and checkpoints are
untouched, and no user data is read (D67 owned synthetic data reused by hash).
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import pexpect

HOST = "192.168.1.104"
TASK = "/home/nik/bc250work/route-port-d69"
PROJECT = "/home/nik/bc250-projects/lora-smoke/project"
ENV_SITE = "/home/nik/bc250-projects/lora-smoke/env/lib/python3.12/site-packages"
D67_DATA = "/home/nik/bc250work/lora-bench-d67/data"
LOCAL_TOOLS = Path(__file__).resolve().parent
BC250 = LOCAL_TOOLS.parent
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/route-port-d69")
EVIDENCE = Path("/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/route-port-d69")
SESSION_TIMEOUT = 620

sys.path.insert(0, str(LOCAL_TOOLS))
import summarize_d69 as summarize  # noqa: E402

# case -> (length, route target or None for reference, max_steps, mode)
CASES = {
    "1024-ref": (1024, None, 6, "steady"),
    "1024-installed": (1024, "installed", 6, "steady"),
    "1024-fused": (1024, "fused", 6, "steady"),
    "4096-ref": (4096, None, 1, "cold"),
    "4096-fused": (4096, "fused", 1, "cold"),
    "8192-fused": (8192, "fused", 1, "cold"),
}


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


def scp_from(sources: list[str], dest: str) -> None:
    subprocess.run(
        ["scp", "-q", *[f"nik@{HOST}:{s}" for s in sources], dest],
        check=True,
        timeout=300,
    )


def app_config(case: str, length: int, max_steps: int) -> dict:
    return {
        "model": "models/qwen3.5-2b",
        "tokenizer": None,
        "train_data": [f"{D67_DATA}/owned-{length}.jsonl"],
        "eval_data": None,
        "output_dir": f"{TASK}/outputs/{case}",
        "resume": None,
        "allow_existing_output": False,
        "validation_split": 0.0,
        "max_samples": 0,
        "seed": 42,
        "max_length": length,
        "length_policy": "trim",
        "epochs": 1.0,
        "max_steps": max_steps,
        "batch_size": 1,
        "eval_batch_size": 1,
        "grad_accum": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "scheduler": "constant",
        "optimizer": "adamw_torch",
        "max_grad_norm": 1.0,
        "rank": 128,
        "alpha": 256,
        "dropout": 0.05,
        "rslora": False,
        "target_modules": "all-linear",
        "precision": "fp16",
        "attention": "eager",
        "gradient_checkpointing": True,
        "packing": False,
        "logging_steps": 1,
        "save_steps": max_steps + 1,
        "save_total_limit": 1,
        "eval_steps": max_steps + 1,
        "report_to": "none",
        "num_workers": 0,
        "trust_remote_code": False,
        "dry_run": False,
    }


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
        "--env",
        "BC250_LORA_INITIAL_SCALE=128",
        "--env",
        "BC250_LORA_BENCHMARK=1",
    ]


def parity_guardian_config() -> dict:
    return {
        "mode": "single",
        "label": "d69-parity",
        "argv": [
            *base_env(),
            "--shm-size",
            "256m",
            f"{TASK}/tools/qualify_route_d69.py",
            "--output",
            f"{TASK}/evidence/d69-parity/qualify.json",
            "--candidate",
            f"{TASK}/tools/bc250_gated_delta.py",
        ],
        "rank": 0,
        "evidence": f"{TASK}/evidence/d69-parity",
        "cwd": PROJECT,
    }


def case_guardian_config(case: str) -> dict:
    length, target, max_steps, _mode = CASES[case]
    argv = [*base_env(), "--shm-size", "256m"]
    if target is None:
        argv += ["--env", "BC250_D69_ROUTE=0"]
    else:
        argv += ["--env", "BC250_D69_ROUTE=1", "--env", f"BC250_D69_NATIVE={target}"]
    argv += [
        f"{TASK}/tools/d69_train_wrapper.py",
        "--config",
        f"{TASK}/configs/{case}.json",
    ]
    return {
        "mode": "single",
        "label": f"d69-{case}",
        "argv": argv,
        "rank": 0,
        "evidence": f"{TASK}/evidence/d69-{case}",
        "cwd": PROJECT,
        "audit": f"{TASK}/outputs/{case}/audit-rank0.json",
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
        final = json.loads(ssh(f"cat {TASK}/evidence/d69-{name}/final.json"))
    except RuntimeError:
        return False
    return final.get("status") == "passed"


def clean_failed(name: str, pin: str) -> None:
    """Remove a failed phase's evidence (root-owned guard subdir) and output
    (the app refuses pre-existing output directories) before re-running."""
    ssh(f"rm -rf {TASK}/outputs/{name}")
    command = f"doas rm -rf {TASK}/evidence/d69-{name}"
    child = spawn_root(command, pin, timeout=60)
    try:
        child.expect(pexpect.EOF, timeout=60)
    finally:
        child.close()
    if child.exitstatus != 0:
        raise RuntimeError(f"cleanup of {name} evidence failed")


def run_guardian(name: str, config: dict, pin: str) -> bool:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_SCRATCH / f"guard-{name}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    scp_to([str(config_path)], f"{TASK}/configs/")
    command = f"doas python3 {TASK}/tools/guard_d69.py {TASK}/configs/guard-{name}.json"
    child = spawn_root(command, pin)
    log = (CONFIG_SCRATCH / f"guard-{name}.log").open("w")
    child.logfile_read = log
    try:
        child.expect(pexpect.EOF, timeout=SESSION_TIMEOUT)
    finally:
        child.close()
        log.close()
    print(f"D69_CASE {name} ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def stage(pin: str) -> None:
    ssh(f"mkdir -p {TASK}/tools {TASK}/configs {TASK}/outputs {TASK}/evidence")
    scp_to(
        [
            str(LOCAL_TOOLS / "guard_d69.py"),
            str(LOCAL_TOOLS / "bounded_launch.py"),
            str(BC250 / "route_qwen35_gated_delta_d69.py"),
            str(BC250 / "d69_train_wrapper.py"),
            str(BC250 / "qualify_route_d69.py"),
            str(
                BC250.parent.parent
                / "sources/bc250-pytorch/torch/nn/bc250_gated_delta.py"
            ),
        ],
        f"{TASK}/tools/",
    )
    print("D69_STAGE tools staged", flush=True)


def verify_data() -> dict:
    manifest = json.loads(ssh(f"cat {D67_DATA}/manifest.json"))
    for length, info in manifest["files"].items():
        digest = ssh(f"sha256sum {info['path']}").split()[0]
        if digest != info["sha256"]:
            raise RuntimeError(f"owned-{length} data hash mismatch")
        print(f"D69_DATA owned-{length} verified {digest[:12]}", flush=True)
    return manifest


def collect(statuses: dict) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    cases = {}
    parity = json.loads(ssh(f"cat {TASK}/evidence/d69-parity/qualify.json"))
    (EVIDENCE / "qualify-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    for name in CASES:
        length, _target, _steps, mode = CASES[name]
        try:
            audit = json.loads(ssh(f"cat {TASK}/outputs/{name}/audit-rank0.json"))
        except RuntimeError:
            # Safely stopped case (e.g. feasibility beyond the child budget):
            # retain the guardian record instead of an audit.
            final = json.loads(ssh(f"cat {TASK}/evidence/d69-{name}/final.json"))
            cases[name] = {
                "mode": mode,
                "status": "stopped",
                "guardian_error": final.get("error"),
                "guardian_status": final.get("status"),
            }
            (EVIDENCE / f"{name}-guardian.json").write_text(
                json.dumps(final, indent=2) + "\n"
            )
            print(f"D69_SUMMARY {name} STOPPED {final.get('error')}", flush=True)
            continue
        route_path = f"{TASK}/outputs/{name}/route-record-rank0.json"
        try:
            route = json.loads(ssh(f"cat {route_path}"))
        except RuntimeError:
            route = None
        (EVIDENCE / f"{name}-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        if route is not None:
            (EVIDENCE / f"{name}-route.json").write_text(
                json.dumps(route, indent=2) + "\n"
            )
        cases[name] = summarize.summarize_case(audit, route, mode)
        cases[name]["status"] = "passed" if statuses.get(name) else "guardian-failed"
        print(f"D69_SUMMARY {name} {json.dumps(cases[name])}", flush=True)
    summary = summarize.build_summary(cases)
    summary["guardian_status"] = statuses
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    pin = recover_pin()
    stage(pin)
    verify_data()
    statuses = {}
    if phase_passed("parity"):
        print("D69_CASE parity SKIP (already passed)", flush=True)
        statuses["parity"] = True
    else:
        if ssh(f"test -d {TASK}/evidence/d69-parity && echo yes || true").strip():
            clean_failed("parity", pin)
        statuses["parity"] = run_guardian("parity", parity_guardian_config(), pin)
    if not statuses["parity"]:
        raise RuntimeError("parity fixture failed; stopping before LoRA cases")
    for name in CASES:
        if phase_passed(name):
            print(f"D69_CASE {name} SKIP (already passed)", flush=True)
            statuses[name] = True
            continue
        if ssh(f"test -d {TASK}/evidence/d69-{name} && echo yes || true").strip():
            clean_failed(name, pin)
        length, _target, max_steps, _mode = CASES[name]
        cfg = app_config(name, length, max_steps)
        cfg_path = CONFIG_SCRATCH / f"app-{name}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        scp_to([str(cfg_path)], f"{TASK}/configs/{name}.json")
        statuses[name] = run_guardian(name, case_guardian_config(name), pin)
    summary = collect(statuses)
    print("D69_RATIOS " + json.dumps(summary["ratios"]), flush=True)
    print("D69_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
