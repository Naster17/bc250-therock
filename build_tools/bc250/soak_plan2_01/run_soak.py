#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-01 workstation orchestrator: production-readiness stability soak.

Accumulates a multi-hour LoRA run as a chain of bounded guardian segments (each
<=460 s, the qualified envelope), joined by EXACT checkpoint/resume. Each segment
trains UPDATES_PER_SEGMENT more optimizer steps (cumulative max_steps), saves a
checkpoint, and the next segment resumes it. This respects the qualified
per-segment envelope AND validates restart reliability. Runs WITHOUT
BC250_LORA_BENCHMARK (benchmark mode refuses resume); the app's Audit callback
still records per-update finite/grad/skip/seconds/peak rows.

Durable host state (soak-state.json) lets the orchestrator resume after a session
interruption: completed segments are not re-run. The D71 `timeout 430` podman
backstop protects every segment's container. Single host (.104); owned D67 data
only; no user application; no default/release change.

Root steps use ssh -tt doas with the PIN recovered at runtime from the Codex
rollout log (never printed, never persisted). Recreate per milestone; delete at
closeout.

Usage:
    python3 run_soak.py --max-segments 2          # pilot (validate chain + timing)
    python3 run_soak.py                            # full soak (resumes pilot)
    python3 run_soak.py --fresh                    # discard state, restart
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pexpect

HOST = "192.168.1.104"
TASK = "/home/nik/bc250work/soak-plan2-01"
PROJECT = "/home/nik/bc250-projects/lora-smoke/project"
ENTRY = f"{PROJECT}/train_two_host.py"
ENV_SITE = "/home/nik/bc250-projects/lora-smoke/env/lib/python3.12/site-packages"
D67_DATA = "/home/nik/bc250work/lora-bench-d67/data"
LOCAL_TOOLS = Path(__file__).resolve().parent
BC250 = LOCAL_TOOLS.parent
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/soak-plan2-01")
EVIDENCE = Path("/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/soak-plan2-01")
SESSION_TIMEOUT = 900
LENGTH = 1024
TARGET_SECONDS = 7200.0
UPDATES_PER_SEGMENT = 10
MAX_SEGMENTS = 40

sys.path.insert(0, str(LOCAL_TOOLS))
import summarize_soak as summarize  # noqa: E402


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


def base_env() -> list[str]:
    # NOTE: deliberately NO BC250_LORA_BENCHMARK — benchmark mode refuses resume.
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
    ]


def app_config(k: int, n: int) -> dict:
    resume = None if k == 0 else f"{TASK}/outputs/seg{k - 1}/checkpoint-{k * n}"
    return {
        "model": "models/qwen3.5-2b",
        "tokenizer": None,
        "train_data": [f"{D67_DATA}/owned-{LENGTH}.jsonl"],
        "eval_data": None,
        "output_dir": f"{TASK}/outputs/seg{k}",
        "resume": resume,
        "allow_existing_output": False,
        "validation_split": 0.0,
        "max_samples": 0,
        "seed": 42,
        "max_length": LENGTH,
        "length_policy": "trim",
        # Large epoch count so the app's epoch-based final-save suppression never
        # fires; save_steps=n saves a resumable checkpoint at each segment end.
        "epochs": 10000.0,
        "max_steps": (k + 1) * n,
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
        "save_steps": n,
        "save_total_limit": 2,
        "eval_steps": (k + 1) * n + 1,
        "report_to": "none",
        "num_workers": 0,
        "trust_remote_code": False,
        "dry_run": False,
    }


def guardian_config(k: int) -> dict:
    return {
        "mode": "single",
        "label": f"soak-seg{k}",
        "argv": [
            *base_env(),
            "--shm-size",
            "256m",
            ENTRY,
            "--config",
            f"{TASK}/configs/seg{k}.json",
        ],
        "rank": 0,
        "evidence": f"{TASK}/evidence/soak-seg{k}",
        "cwd": PROJECT,
        "audit": f"{TASK}/outputs/seg{k}/audit-rank0.json",
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


def load_state() -> dict:
    try:
        return json.loads(ssh(f"cat {TASK}/soak-state.json"))
    except RuntimeError:
        return {
            "completed_segments": 0,
            "accumulated_seconds": 0.0,
            "segments": [],
            "telemetry": [],
        }


def save_state(state: dict) -> None:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    local = CONFIG_SCRATCH / "soak-state.json"
    local.write_text(json.dumps(state, indent=2) + "\n")
    scp_to([str(local)], f"{TASK}/soak-state.json")


def run_guardian(k: int, pin: str) -> bool:
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    config = guardian_config(k)
    config_path = CONFIG_SCRATCH / f"guard-seg{k}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    scp_to([str(config_path)], f"{TASK}/configs/")
    command = (
        f"doas python3 {TASK}/tools/guard_soak.py {TASK}/configs/guard-seg{k}.json"
    )
    child = spawn_root(command, pin)
    log = (CONFIG_SCRATCH / f"guard-seg{k}.log").open("w")
    child.logfile_read = log
    try:
        child.expect(pexpect.EOF, timeout=SESSION_TIMEOUT)
    finally:
        child.close()
        log.close()
    print(f"SOAK_SEG {k} ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def clean_segment(k: int, pin: str) -> None:
    """Remove an interrupted segment's evidence + output so it can re-run fresh."""
    child = spawn_root(
        f"doas rm -rf {TASK}/evidence/soak-seg{k} {TASK}/outputs/seg{k}",
        pin,
        timeout=60,
    )
    try:
        child.expect(pexpect.EOF, timeout=60)
    finally:
        child.close()
    if child.exitstatus != 0:
        raise RuntimeError(f"cleanup of interrupted segment {k} failed")


def sample_telemetry(k: int) -> dict:
    temp = ssh("cat /sys/class/hwmon/hwmon*/temp1_input 2>/dev/null | head -1").strip()
    freq = ssh("cat /sys/class/hwmon/hwmon*/freq1_input 2>/dev/null | head -1").strip()
    boot = ssh("cat /proc/sys/kernel/random/boot_id").strip()
    return {
        "segment": k,
        "temp_c": int(temp) / 1000.0 if temp else None,
        "freq_hz": int(freq) if freq else None,
        "boot_id": boot,
    }


def idle_check() -> None:
    foreign = ssh("podman ps -q").split()
    if foreign:
        raise RuntimeError(f"board not idle; foreign containers: {foreign}")
    print(
        f"SOAK_IDLE boot_id={ssh('cat /proc/sys/kernel/random/boot_id').strip()} containers=0",
        flush=True,
    )


def stage() -> None:
    ssh(f"mkdir -p {TASK}/tools {TASK}/configs {TASK}/outputs {TASK}/evidence")
    scp_to(
        [
            str(LOCAL_TOOLS / "guard_soak.py"),
            str(LOCAL_TOOLS / "bounded_launch.py"),
            str(LOCAL_TOOLS / "kernel_noise.py"),
        ],
        f"{TASK}/tools/",
    )
    print("SOAK_STAGE tools staged", flush=True)


def verify_data() -> None:
    manifest = json.loads(ssh(f"cat {D67_DATA}/manifest.json"))
    info = manifest["files"][str(LENGTH)]
    digest = ssh(f"sha256sum {info['path']}").split()[0]
    if digest != info["sha256"]:
        raise RuntimeError(f"owned-{LENGTH} data hash mismatch")
    print(f"SOAK_DATA owned-{LENGTH} verified {digest[:12]}", flush=True)


def collect(state: dict, statuses: dict) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    audits = []
    for k in range(state["completed_segments"]):
        audit = json.loads(ssh(f"cat {TASK}/outputs/seg{k}/audit-rank0.json"))
        audits.append(audit)
        (EVIDENCE / f"seg{k}-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        final = json.loads(ssh(f"cat {TASK}/evidence/soak-seg{k}/final.json"))
        (EVIDENCE / f"seg{k}-guardian.json").write_text(
            json.dumps(final, indent=2) + "\n"
        )
    (EVIDENCE / "telemetry.json").write_text(
        json.dumps(state["telemetry"], indent=2) + "\n"
    )
    summary = summarize.build_summary(
        audits, state["telemetry"], TARGET_SECONDS, statuses
    )
    summary["updates_per_segment"] = UPDATES_PER_SEGMENT
    summary["length"] = LENGTH
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {}
    for path in sorted(EVIDENCE.glob("*.json")):
        if path.name == "artifacts.json":
            continue
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (EVIDENCE / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> int:
    global TARGET_SECONDS, UPDATES_PER_SEGMENT, MAX_SEGMENTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-seconds", type=float, default=TARGET_SECONDS)
    parser.add_argument("--updates-per-segment", type=int, default=UPDATES_PER_SEGMENT)
    parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    TARGET_SECONDS = args.target_seconds
    UPDATES_PER_SEGMENT = args.updates_per_segment
    MAX_SEGMENTS = args.max_segments

    pin = recover_pin()
    idle_check()
    stage()
    verify_data()
    if args.fresh:
        ssh(f"rm -f {TASK}/soak-state.json")
    state = load_state()
    print(
        f"SOAK_BOUNDS target={TARGET_SECONDS}s updates/seg={UPDATES_PER_SEGMENT} "
        f"max_segments={MAX_SEGMENTS} drift=[{summarize.DRIFT_MIN},{summarize.DRIFT_MAX}] "
        f"mem<={summarize.MEMORY_RATIO_MAX} temp<={summarize.TEMP_MAX_C}C",
        flush=True,
    )
    print(
        f"SOAK_RESUME completed={state['completed_segments']} "
        f"accumulated={state['accumulated_seconds']:.1f}s",
        flush=True,
    )
    statuses = {f"seg{s['k']}": True for s in state["segments"]}
    k = state["completed_segments"]
    while state["accumulated_seconds"] < TARGET_SECONDS and k < MAX_SEGMENTS:
        cfg = app_config(k, UPDATES_PER_SEGMENT)
        if cfg["resume"]:
            if not ssh(f"test -d {cfg['resume']} && echo yes || true").strip():
                raise RuntimeError(f"resume checkpoint missing: {cfg['resume']}")
        cfg_path = CONFIG_SCRATCH / f"app-seg{k}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        scp_to([str(cfg_path)], f"{TASK}/configs/seg{k}.json")
        if ssh(f"test -d {TASK}/evidence/soak-seg{k} && echo yes || true").strip():
            clean_segment(k, pin)  # leftover from an interrupted attempt
        passed = run_guardian(k, pin)
        statuses[f"seg{k}"] = passed
        if not passed:
            print(f"SOAK_SEG {k} FAILED; stopping (evidence retained)", flush=True)
            break
        audit = json.loads(ssh(f"cat {TASK}/outputs/seg{k}/audit-rank0.json"))
        seg_seconds = sum(float(r["seconds"]) for r in audit["rows"])
        ckpt = f"{TASK}/outputs/seg{k}/checkpoint-{(k + 1) * UPDATES_PER_SEGMENT}"
        if not ssh(f"test -d {ckpt} && echo yes || true").strip():
            raise RuntimeError(f"segment-end checkpoint missing: {ckpt}")
        telemetry = sample_telemetry(k)
        state["segments"].append(
            {"k": k, "seconds": round(seg_seconds, 3), "updates": len(audit["rows"])}
        )
        state["telemetry"].append(telemetry)
        state["accumulated_seconds"] += seg_seconds
        state["completed_segments"] = k + 1
        save_state(state)
        print(
            f"SOAK_PROGRESS seg={k} updates={len(audit['rows'])} "
            f"seg_s={seg_seconds:.1f} accumulated={state['accumulated_seconds']:.1f}s "
            f"({state['accumulated_seconds'] / 3600:.2f}h) temp={telemetry['temp_c']}C",
            flush=True,
        )
        k += 1
    summary = collect(state, statuses)
    print(
        f"SOAK_RESULT status={summary['status']} segments={summary['segments']} "
        f"updates={summary['total_updates']} hours={summary['accumulated_hours']} "
        f"drift={summary['drift_ratio']} mem={summary['memory_ratio']} "
        f"temp_max={summary['max_temp_c']}",
        flush=True,
    )
    print("SOAK_CHECKS " + json.dumps(summary["checks"]), flush=True)
    print("SOAK_DONE", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
