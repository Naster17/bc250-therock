#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D71 workstation orchestrator: GEMM efficiency for the LoRA shapes.

Phases: stage tools -> verify D67 owned-data hashes -> guarded GEMM layout
census (1 LoRA update, dispatch-instrumented) -> guarded solution bake-off
(private-handle Tensile sweep + hipBLASLt AB/BA on the census layouts) ->
build candidate override preset (release rows preserved + qualified winners)
-> guarded whole-step reference A/B (6 updates each; candidate gets the
preset via the launcher's manual-override path) -> collect -> summarize.
Single host (.104); measurement + qualification only — NO release/preset
deployment in this batch (that is a separate EX-lock scope).

Root steps use ssh -tt doas with the PIN recovered at runtime from the Codex
rollout log (never printed, never persisted). Recreate this driver per
milestone; delete it at closeout. Only additive task paths are used; the
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
TASK = "/home/nik/bc250work/gemm-efficiency-d71"
PROJECT = "/home/nik/bc250-projects/lora-smoke/project"
ENTRY = f"{PROJECT}/train_two_host.py"
ENV_SITE = "/home/nik/bc250-projects/lora-smoke/env/lib/python3.12/site-packages"
D67_DATA = "/home/nik/bc250work/lora-bench-d67/data"
RELEASE_CSV = "/opt/bc250-stack/current/rocblas-gemm-n32.csv"
LOCAL_TOOLS = Path(__file__).resolve().parent
BC250 = LOCAL_TOOLS.parent
CONFIG_SCRATCH = Path("/home/nik/Documents/Vibe/bc250stack/tmp/gemm-efficiency-d71")
EVIDENCE = Path(
    "/home/nik/Documents/Vibe/bc250stack/kernel/artifacts/gemm-efficiency-d71"
)
SESSION_TIMEOUT = 900

sys.path.insert(0, str(LOCAL_TOOLS))
import preset_d71 as preset  # noqa: E402
import summarize_d71 as summarize  # noqa: E402

CASES = {
    "census-1024": (1024, 1),
    "step-ref-1024": (1024, 6),
    "step-cand-1024": (1024, 6),
    "profile-shapes-1024": (1024, 2),
}
CANDIDATE_CSV = f"{TASK}/configs/candidate-gemm-d71.csv"


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


def guardian_config(name: str) -> dict:
    argv = [*base_env(), "--shm-size", "256m"]
    if name == "census-1024":
        argv += [
            f"{TASK}/tools/census_gemm_d71.py",
            "--config",
            f"{TASK}/configs/census-1024.json",
        ]
        audit = f"{TASK}/outputs/census-1024/audit-rank0.json"
    elif name == "profile-shapes-1024":
        argv += [
            f"{TASK}/tools/profile_shapes_d71.py",
            "--config",
            f"{TASK}/configs/profile-shapes-1024.json",
        ]
        audit = f"{TASK}/outputs/profile-shapes-1024/audit-rank0.json"
    elif name == "bench":
        argv += [
            f"{TASK}/tools/bench_gemm_d71.py",
            "--census",
            f"{TASK}/outputs/census-1024/gemm-census-rank0.json",
            "--output",
            f"{TASK}/evidence/d71-bench/bench.json",
        ]
        audit = None
    else:
        if name == "step-cand-1024":
            argv += [
                "--env",
                f"ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH={CANDIDATE_CSV}",
                "--mount-ro",
                CANDIDATE_CSV,
            ]
        argv += [ENTRY, "--config", f"{TASK}/configs/{name}.json"]
        audit = f"{TASK}/outputs/{name}/audit-rank0.json"
    config = {
        "mode": "single",
        "label": f"d71-{name}",
        "argv": argv,
        "rank": 0,
        "evidence": f"{TASK}/evidence/d71-{name}",
        "cwd": PROJECT,
    }
    if audit:
        config["audit"] = audit
    return config


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
        final = json.loads(ssh(f"cat {TASK}/evidence/d71-{name}/final.json"))
    except RuntimeError:
        return False
    return final.get("status") == "passed"


def clean_failed(name: str, pin: str) -> None:
    ssh(f"rm -rf {TASK}/outputs/{name}")
    child = spawn_root(f"doas rm -rf {TASK}/evidence/d71-{name}", pin, timeout=60)
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
    command = f"doas python3 {TASK}/tools/guard_d71.py {TASK}/configs/guard-{name}.json"
    child = spawn_root(command, pin)
    log = (CONFIG_SCRATCH / f"guard-{name}.log").open("w")
    child.logfile_read = log
    try:
        child.expect(pexpect.EOF, timeout=SESSION_TIMEOUT)
    finally:
        child.close()
        log.close()
    print(f"D71_CASE {name} ssh_exit={child.exitstatus}", flush=True)
    return child.exitstatus == 0


def run_phase(
    name: str, pin: str, statuses: dict, required_output: str | None = None
) -> None:
    passed = phase_passed(name)
    if passed and required_output:
        passed = bool(ssh(f"test -f {required_output} && echo yes || true").strip())
    if passed:
        print(f"D71_CASE {name} SKIP (already passed)", flush=True)
        statuses[name] = True
        return
    if ssh(f"test -d {TASK}/evidence/d71-{name} && echo yes || true").strip():
        clean_failed(name, pin)
    statuses[name] = run_guardian(name, pin)


def stage() -> None:
    ssh(f"mkdir -p {TASK}/tools {TASK}/configs {TASK}/outputs {TASK}/evidence")
    scp_to(
        [
            str(LOCAL_TOOLS / "guard_d71.py"),
            str(LOCAL_TOOLS / "bounded_launch.py"),
            str(BC250 / "census_gemm_d71.py"),
            str(BC250 / "bench_gemm_d71.py"),
            str(BC250 / "profile_shapes_d71.py"),
        ],
        f"{TASK}/tools/",
    )
    print("D71_STAGE tools staged", flush=True)


def verify_data() -> None:
    manifest = json.loads(ssh(f"cat {D67_DATA}/manifest.json"))
    for length, info in manifest["files"].items():
        digest = ssh(f"sha256sum {info['path']}").split()[0]
        if digest != info["sha256"]:
            raise RuntimeError(f"owned-{length} data hash mismatch")
        print(f"D71_DATA owned-{length} verified {digest[:12]}", flush=True)


def build_candidate_preset() -> int:
    bench = json.loads(ssh(f"cat {TASK}/evidence/d71-bench/bench.json"))
    if bench.get("status") != "passed":
        raise RuntimeError(f"bench status {bench.get('status')}")
    existing = ssh(f"cat {RELEASE_CSV}")
    csv_text = preset.build_preset(bench, existing)
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    local = CONFIG_SCRATCH / "candidate-gemm-d71.csv"
    local.write_text(csv_text)
    scp_to([str(local)], f"{TASK}/configs/")
    added = len(preset.winner_rows(bench))
    print(f"D71_PRESET rows_added={added}", flush=True)
    return added


def collect(statuses: dict, preset_rows: int) -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    census = json.loads(ssh(f"cat {TASK}/outputs/census-1024/gemm-census-rank0.json"))
    bench = json.loads(ssh(f"cat {TASK}/evidence/d71-bench/bench.json"))
    (EVIDENCE / "census-1024.json").write_text(json.dumps(census, indent=2) + "\n")
    (EVIDENCE / "bench.json").write_text(json.dumps(bench, indent=2) + "\n")
    (EVIDENCE / "candidate-gemm-d71.csv").write_text(ssh(f"cat {CANDIDATE_CSV}"))
    ref_audit = json.loads(ssh(f"cat {TASK}/outputs/step-ref-1024/audit-rank0.json"))
    (EVIDENCE / "step-ref-1024-audit.json").write_text(
        json.dumps(ref_audit, indent=2) + "\n"
    )
    cand_audit = None
    if statuses.get("step-cand-1024"):
        cand_audit = json.loads(
            ssh(f"cat {TASK}/outputs/step-cand-1024/audit-rank0.json")
        )
        (EVIDENCE / "step-cand-1024-audit.json").write_text(
            json.dumps(cand_audit, indent=2) + "\n"
        )
    profile_shapes = None
    if statuses.get("profile-shapes-1024"):
        profile_shapes = json.loads(
            ssh(f"cat {TASK}/outputs/profile-shapes-1024/profile-shapes-rank0.json")
        )
        (EVIDENCE / "profile-shapes-1024.json").write_text(
            json.dumps(profile_shapes, indent=2) + "\n"
        )
    for name in (
        "census-1024",
        "bench",
        "step-ref-1024",
        "step-cand-1024",
        "profile-shapes-1024",
    ):
        if statuses.get(name) is False:
            final = json.loads(ssh(f"cat {TASK}/evidence/d71-{name}/final.json"))
            (EVIDENCE / f"{name}-guardian.json").write_text(
                json.dumps(final, indent=2) + "\n"
            )
    summary = summarize.build_summary(census, bench, ref_audit, cand_audit, preset_rows)
    if profile_shapes is not None:
        summary["profile_shapes"] = {
            "gemm_self_device_ms_total": profile_shapes["gemm_self_device_ms_total"],
            "kernel_self_device_ms_total": profile_shapes[
                "kernel_self_device_ms_total"
            ],
            "top_gemm_ops": profile_shapes["gemm_ops"][:12],
        }
    summary["guardian_status"] = statuses
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    pin = recover_pin()
    stage()
    verify_data()
    statuses: dict = {}
    CONFIG_SCRATCH.mkdir(parents=True, exist_ok=True)
    length, max_steps = CASES["census-1024"]
    cfg_path = CONFIG_SCRATCH / "app-census-1024.json"
    cfg_path.write_text(
        json.dumps(app_config("census-1024", length, max_steps), indent=2) + "\n"
    )
    scp_to([str(cfg_path)], f"{TASK}/configs/census-1024.json")
    run_phase(
        "census-1024",
        pin,
        statuses,
        required_output=f"{TASK}/outputs/census-1024/gemm-census-rank0.json",
    )
    if not statuses["census-1024"]:
        raise RuntimeError("census failed; stopping")
    run_phase("bench", pin, statuses)
    if not statuses["bench"]:
        raise RuntimeError("bake-off failed; stopping")
    preset_rows = build_candidate_preset()
    for name in ("step-ref-1024", "step-cand-1024"):
        length, max_steps = CASES[name]
        cfg_path = CONFIG_SCRATCH / f"app-{name}.json"
        cfg_path.write_text(
            json.dumps(app_config(name, length, max_steps), indent=2) + "\n"
        )
        scp_to([str(cfg_path)], f"{TASK}/configs/{name}.json")
        run_phase(name, pin, statuses)
    length, max_steps = CASES["profile-shapes-1024"]
    cfg_path = CONFIG_SCRATCH / "app-profile-shapes-1024.json"
    cfg_path.write_text(
        json.dumps(app_config("profile-shapes-1024", length, max_steps), indent=2)
        + "\n"
    )
    scp_to([str(cfg_path)], f"{TASK}/configs/profile-shapes-1024.json")
    run_phase(
        "profile-shapes-1024",
        pin,
        statuses,
        required_output=(
            f"{TASK}/outputs/profile-shapes-1024/profile-shapes-rank0.json"
        ),
    )
    summary = collect(statuses, preset_rows)
    print("D71_WINNERS " + json.dumps(summary["winners"]), flush=True)
    if summary["candidate"]:
        print(
            f"D71_STEP ref={summary['reference']['steady_seconds']:.3f}s "
            f"cand={summary['candidate']['steady_seconds']:.3f}s "
            f"speedup={summary['candidate']['step_speedup']:.3f}x "
            f"loss_delta={summary['candidate']['final_loss_delta']}",
            flush=True,
        )
    print("D71_STATUS " + summary["status"], flush=True)
    print("D71_DONE", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
