#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install the qualified shared visibility refusal on .104 under both EX locks."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from board_preflight import preflight

WORK = Path("/home/nik/bc250work")
SOURCE = WORK / "tools/visibility-d44/bc250_torch.py"
ROOT = WORK / "evidence/visibility-d44-install-r1"
OLD = "c5167f171eefb6df728eea2c1063f2af1ca1317a03ea8d69ab67f36a700f6106"
TARGETS = [
    Path("/usr/local/bin/bc250-torch"),
    Path("/opt/bc250-stack/tools/distributed-d09/bc250_torch.py"),
]


def as_user():
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    proof = json.loads((WORK / "evidence/visibility-d44-r2/result.json").read_text())
    assert proof["status"] == "passed" and proof["guard"]["status"] == "acknowledged"
    data = SOURCE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert proof["sources"][SOURCE.name] == digest
    ROOT.mkdir(exist_ok=False)
    with (WORK / "evidence/.board-reservation.lock").open("a") as board, Path(
        "/opt/bc250-stack/current/.runtime.lock"
    ).open("r") as runtime:
        fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runtime, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = preflight("d44-exclusive-install")
        for index, target in enumerate(TARGETS):
            assert not target.is_symlink()
            assert hashlib.sha256(target.read_bytes()).hexdigest() == OLD
            shutil.copy2(target, ROOT / f"{index}-bc250_torch.py")
        records = []
        try:
            for target in TARGETS:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False
                ) as stream:
                    pending = Path(stream.name)
                    stream.write(data)
                pending.chmod(target.stat().st_mode & 0o777)
                pending.replace(target)
            for name, command, expected in [
                (
                    "torch",
                    [
                        "/usr/local/bin/bc250-torch",
                        "-c",
                        "raise RuntimeError('WORKER_MUST_NOT_RUN')",
                    ],
                    2,
                ),
                (
                    "train",
                    [
                        "/usr/local/bin/bc250-train",
                        "-c",
                        "raise RuntimeError('WORKER_MUST_NOT_RUN')",
                    ],
                    2,
                ),
                (
                    "distributed",
                    [
                        "/usr/local/bin/bc250-distributed",
                        "--node-rank",
                        "0",
                        str(WORK / "tools/visibility-d44/visibility_d44_probe.py"),
                    ],
                    1,
                ),
            ]:
                env = dict(
                    os.environ,
                    HIP_VISIBLE_DEVICES="-1",
                    HOME="/home/nik",
                    USER="nik",
                    LOGNAME="nik",
                )
                result = subprocess.run(
                    command,
                    env=env,
                    preexec_fn=as_user,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                record = dict(
                    name=name,
                    command=command,
                    exit=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                records.append(record)
                assert result.returncode == expected, record
                assert (
                    "recorded ROCr import crash" in result.stderr
                    and "WORKER_MUST_NOT_RUN" not in result.stderr
                ), record
            after = preflight("d44-installed-refusals")
        except BaseException:
            for index, target in enumerate(TARGETS):
                shutil.copy2(ROOT / f"{index}-bc250_torch.py", target)
            (ROOT / "rollback.json").write_text(
                json.dumps(dict(status="rolled-back", checks=records), indent=2)
            )
            raise
        (ROOT / "result.json").write_text(
            json.dumps(
                dict(
                    status="passed",
                    source_sha256=digest,
                    old_sha256=OLD,
                    targets=[str(p) for p in TARGETS],
                    checks=records,
                    before=before,
                    after=after,
                    lock="EX board reservation + EX daily runtime",
                    native_binaries_changed=False,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
