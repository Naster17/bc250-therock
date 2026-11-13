#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D70 workstation orchestrator: LoRA step decomposition + launch-reduction
feasibility probe.

Phases: stage tools -> verify D67 owned-data hashes (read-only reuse) ->
guarded feasibility probe (profiler / HIP graphs / instrumentation selftest)
-> guarded instrumented LoRA decomposition at 1024 (6 updates, reference path)
-> guarded one-update kernel census at 1024 (only if the probe found
torch.profiler CUDA activity working) -> collect evidence -> summarize.
Single host (.104); measurement only, no source/default/release change.

Root steps use ssh -tt doas with the PIN recovered at runtime from the Codex
rollout log (never printed, never persisted). Recreate this driver per
milestone; delete it at closeout. Only additive task paths are used
(/home/nik/bc250work/launch-reduction-d70 and fresh output directories); the
application's original entry points, completed outputs and checkpoints are
untouched, and no user data is read (D67 owned synthetic data reused by hash).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
import re
import subprocess
import sys

import pexpect

HOST = "192.168.1.104"
TASK = "/home/nik/bc250work/launch-reduction-d70"
PROJECT = "/home/nik/bc250-projects/lora-smoke/project"
ENV_SITE = "/home/nik/bc250-projects/lora-smoke/env/lib/python3.12/site-packages"
D67_DATA = "/home/nik/bc250work/lora-bench-d67/data"
LOCAL_TOOLS = Path(__file__).resolve().parent
BC250 = LOCAL_TOOLS.parent
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/launch-reduction-d70")
EVIDENCE = Path(
    "/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/launch-reduction-d70"
)
SESSION_TIMEOUT = 620

sys.path.insert(0, str(LOCAL_TOOLS))
import summarize_d70 as summarize  # noqa: E402

# case -> (length, max_steps, profile)
CASES = {
    "decomp-1024": (1024, 6, False),
    "profile-1024": (1024, 2, True),
    # One cold update at 4096: attributes the superlinear cost growth D67/D69
    # measured (324 s ~ 9x the 1024 step, beyond O(L) GEMM scaling). D69 ran
    # one 4096 update inside the same 460 s child budget.
    "decomp-4096": (4096, 1, False),
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


def probe_guardian_config() -> dict:
    return {
        "mode": "single",
        "label": "d70-probe",
        "argv": [
            *base_env(),
            "--shm-size",
            "256m",
            f"{TASK}/tools/probe_launch_d70.py",
            "--output",
            f"{TASK}/evidence/d70-probe/probe.json",
            "--candidate",
            f"{TASK}/tools/bc250_gated_delta.py",
        ],
        "rank": 0,
        "evidence": f"{TASK}/evidence/d70-probe",
        "cwd": PROJECT,
    }


def case_guardian_config(case: str) -> dict:
    length, max_steps, profile = CASES[case]
    argv = [*base_env(), "--shm-size", "256m"]
    if profile:
        argv += ["--env", "BC250_D70_PROFILE=1"]
    argv += [
        f"{TASK}/tools/decompose_lora_d70.py",
        "--config",
        f"{TASK}/configs/{case}.json",
    ]
    return {
        "mode": "single",
        "label": f"d70-{case}",
        "argv": argv,
        "rank": 0,
        "evidence": f"{TASK}/evidence/d70-{case}",
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
        final = json.loads(ssh(f"cat {TASK}/evidence/d70-{name}/final.json"))
    except RuntimeError:
        return False
    return final.get("status") == "passed"


def clean_failed(name: str, pin: str) -> None:
    """Remove a failed phase's evidence (root-owned guard subdir) and output
    (the app refuses pre-existing output directories) before re-running."""
    ssh(f"rm -rf {TASK}/outputs/{name}")
    command = f"doas rm -rf {TASK}/evidence/d70-{name}"
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
    command = f"doas python3 {TASK}/tools/guard_d70.py {TASK}/configs/guard-{name}.json"
    child = spawn_root(command, pin)
    log = (CONFIG_SCRATCH / f"guard-{name}.log").open("w")
    child.logfile_read = log
    try:
        child.expect(pexpect.EOF, timeout=SESSION_TIMEOUT)
    finally:
        child.close()
        log.close()
    print(f"D70_CASE {name} ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def stage() -> None:
    ssh(f"mkdir -p {TASK}/tools {TASK}/configs {TASK}/outputs {TASK}/evidence")
    scp_to(
        [
            str(LOCAL_TOOLS / "guard_d70.py"),
            str(LOCAL_TOOLS / "bounded_launch.py"),
            str(BC250 / "decompose_lora_d70.py"),
            str(BC250 / "probe_launch_d70.py"),
            str(BC250 / "route_qwen35_gated_delta_d69.py"),
            str(
                BC250.parent.parent
                / "sources/bc250-pytorch/torch/nn/bc250_gated_delta.py"
            ),
        ],
        f"{TASK}/tools/",
    )
    print("D70_STAGE tools staged", flush=True)


def verify_data() -> dict:
    manifest = json.loads(ssh(f"cat {D67_DATA}/manifest.json"))
    for length, info in manifest["files"].items():
        digest = ssh(f"sha256sum {info['path']}").split()[0]
        if digest != info["sha256"]:
            raise RuntimeError(f"owned-{length} data hash mismatch")
        print(f"D70_DATA owned-{length} verified {digest[:12]}", flush=True)
    return manifest


def collect(statuses: dict, probe: dict) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    decomps = {}
    for name in ("decomp-1024", "decomp-4096"):
        if not statuses.get(name):
            if statuses.get(name) is False:
                final = json.loads(ssh(f"cat {TASK}/evidence/d70-{name}/final.json"))
                (EVIDENCE / f"{name}-guardian.json").write_text(
                    json.dumps(final, indent=2) + "\n"
                )
            continue
        decomp = json.loads(ssh(f"cat {TASK}/outputs/{name}/decomposition-rank0.json"))
        audit = json.loads(ssh(f"cat {TASK}/outputs/{name}/audit-rank0.json"))
        (EVIDENCE / f"{name}.json").write_text(json.dumps(decomp, indent=2) + "\n")
        (EVIDENCE / f"{name}-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        decomps[name] = decomp
    decomp = decomps["decomp-1024"]
    profile = None
    if statuses.get("profile-1024"):
        profile = json.loads(ssh(f"cat {TASK}/outputs/profile-1024/profile-rank0.json"))
        profile_audit = json.loads(
            ssh(f"cat {TASK}/outputs/profile-1024/audit-rank0.json")
        )
        (EVIDENCE / "profile-1024.json").write_text(
            json.dumps(profile, indent=2) + "\n"
        )
        (EVIDENCE / "profile-1024-audit.json").write_text(
            json.dumps(profile_audit, indent=2) + "\n"
        )
    elif statuses.get("profile-1024") is False:
        (EVIDENCE / "profile-1024-skip.json").write_text(
            json.dumps(
                {"skipped": True, "reason": "profiler unavailable or case failed"},
                indent=2,
            )
            + "\n"
        )
    summary = summarize.build_summary(
        probe, decomp, profile, decomps.get("decomp-4096")
    )
    summary["guardian_status"] = statuses
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    pin = recover_pin()
    stage()
    verify_data()
    statuses = {}
    if phase_passed("probe"):
        print("D70_CASE probe SKIP (already passed)", flush=True)
        statuses["probe"] = True
    else:
        if ssh(f"test -d {TASK}/evidence/d70-probe && echo yes || true").strip():
            clean_failed("probe", pin)
        statuses["probe"] = run_guardian("probe", probe_guardian_config(), pin)
    if not statuses["probe"]:
        raise RuntimeError("feasibility probe failed; stopping before LoRA cases")
    probe = json.loads(ssh(f"cat {TASK}/evidence/d70-probe/probe.json"))
    if probe.get("status") != "completed":
        raise RuntimeError(f"probe fixture status {probe.get('status')}")
    for name in CASES:
        if name == "profile-1024" and probe["verdicts"].get("profiler") != "works":
            print("D70_CASE profile-1024 SKIP (profiler unavailable)", flush=True)
            statuses[name] = None
            continue
        if phase_passed(name):
            print(f"D70_CASE {name} SKIP (already passed)", flush=True)
            statuses[name] = True
            continue
        if ssh(f"test -d {TASK}/evidence/d70-{name} && echo yes || true").strip():
            clean_failed(name, pin)
        length, max_steps, _profile = CASES[name]
        cfg = app_config(name, length, max_steps)
        cfg_path = CONFIG_SCRATCH / f"app-{name}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        scp_to([str(cfg_path)], f"{TASK}/configs/{name}.json")
        statuses[name] = run_guardian(name, case_guardian_config(name), pin)
    summary = collect(statuses, probe)
    print(
        "D70_SHARES " + json.dumps(summary["reproduction"]["steady_shares"]), flush=True
    )
    print("D70_VERDICTS " + json.dumps(summary["probe_verdicts"]), flush=True)
    print("D70_STATUS " + summary["status"], flush=True)
    print("D70_DONE", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
