#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""D42 isolated deadline bound; same qualified recovery algorithm, original untouched."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="7.2.10-bc250rocm-r3")
    parser.add_argument("--phase", default="boot")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--directory", type=Path, default=Path("/run/bc250-r3-guard"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if os.uname().release != args.release:
        print("R3_GUARD_INACTIVE_RELEASE", flush=True)
        return 0
    if not args.dry_run and os.geteuid() != 0:
        raise RuntimeError("Live recovery requires root")
    if args.timeout < 1 or args.timeout > 1800:
        raise ValueError("Deadline must be within1..1800 seconds")
    args.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.directory.chmod(0o700)
    state_file = args.directory / "state.json"
    ack_file = args.directory / "ack.json"
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    nonce = secrets.token_hex(16)
    deadline = time.monotonic() + args.timeout
    state = dict(
        release=args.release,
        phase=args.phase,
        boot_id=boot_id,
        nonce=nonce,
        timeout_seconds=args.timeout,
        status="armed",
        deadline_monotonic=deadline,
        dry_run=args.dry_run,
    )

    def save() -> None:
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_file)

    save()
    print("R3_GUARD_ARMED", args.phase, args.timeout, flush=True)
    while time.monotonic() < deadline:
        try:
            ack = json.loads(ack_file.read_text())
            if all(ack.get(key) == state[key] for key in ("boot_id", "nonce", "phase")):
                state["status"] = "acknowledged"
                save()
                print("R3_GUARD_ACKNOWLEDGED", args.phase, flush=True)
                return 0
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    state["status"] = "expired"
    save()
    print("R3_GUARD_EXPIRED", args.phase, flush=True)
    if args.dry_run:
        return 2
    # A normal reboot can block in a wedged GPU shutdown callback. Give sync
    # a bounded opportunity; never wait indefinitely for an uninterruptible task.
    sync = subprocess.Popen(["sync"])
    try:
        sync.wait(timeout=2)
    except subprocess.TimeoutExpired:
        sync.kill()
    # Deliberate emergency restart: only the candidate and an unacknowledged
    # deadline reach here. Persistent GRUB default remains the verified r2 entry.
    Path("/proc/sysrq-trigger").write_text("b")
    raise RuntimeError("Emergency restart returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
