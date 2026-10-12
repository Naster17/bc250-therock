#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise installed launcher file flows, checkpoints and container restarts."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import tempfile
import threading


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError("Run host orchestration as root")
    workspace = args.workspace.resolve()
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    uid, gid = manifest["uid"], manifest["gid"]
    root = workspace / "launcher-io-validation"
    root.mkdir(exist_ok=False)
    project = root / "training project"
    project.mkdir()
    logs = workspace / "logs/launcher-io"
    logs.mkdir(exist_ok=True)
    launcher = "/usr/local/bin/bc250-torch"
    checks = []

    def run(
        name: str, arguments: list[str], text: str | None = None, expected: int = 0
    ) -> None:
        result = subprocess.run(
            [launcher, *arguments],
            cwd=project,
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=240,
        )
        (logs / f"{name}.log").write_text(result.stdout)
        if result.returncode != expected:
            raise RuntimeError(
                f"Launcher case {name} failed: {result.returncode}; see {logs / (name + '.log')}"
            )
        checks.append(name)
        print(f"LAUNCHER_CASE_PASS {name}", flush=True)

    with tempfile.TemporaryDirectory(prefix="bc250-launcher-external-") as directory:
        external = Path(directory)
        scripts = external / "external scripts"
        scripts.mkdir()
        shutil.copy2(
            Path(__file__).with_name("validate_launcher_io.py"),
            scripts / "training fixture.py",
        )
        (scripts / "launcher_sidecar.py").write_text("SIDECAR_VALUE = 1704\n")
        data = external / "dataset file.txt"
        data.write_text("host dataset\n")
        payload = external / "download.bin"
        payload.write_bytes(bytes(range(256)) * 1027)
        (project / "linked dataset").symlink_to(data)
        (project / "launcher_module.py").write_text(
            "from pathlib import Path\nassert Path('linked dataset').read_text() == 'host dataset\\n'\nprint('MODULE_IMPORT_PASS')\n"
        )
        (root / "training.env").write_text(
            "CUSTOM_TRAINING_SETTING=value with spaces=42\n"
        )
        for parent in (external, root):
            for path in [parent, *parent.rglob("*")]:
                os.chown(path, uid, gid, follow_symlinks=False)
        handler = partial(QuietHandler, directory=str(external))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        environment = [
            "--env",
            "HF_TOKEN=BC250_LAUNCHER_FAKE_TOKEN",
            "--env",
            f"EXPECTED_HOME={pwd.getpwuid(uid).pw_dir}",
            "--env-file",
            str(root / "training.env"),
        ]
        fixture = [
            "--mount-ro",
            str(data),
            *environment,
            str(scripts / "training fixture.py"),
            "--root",
            str(root),
            "--data",
            str(data),
            "--url",
            f"http://127.0.0.1:{server.server_port}/download.bin",
        ]
        try:
            run("checkpoint-write-download-workers", [*fixture, "--phase", "write"])
            checkpoint = project / "checkpoints with spaces/model checkpoint.pt"
            assert checkpoint.stat().st_uid == uid
            assert checkpoint.stat().st_gid == gid
            (root / "host-added.txt").write_text("visible after container restart\n")
            os.chown(root / "host-added.txt", uid, gid)
            run("checkpoint-resume-host-update", [*fixture, "--phase", "resume"])
            run("module", ["--mount-ro", str(data), "-m", "launcher_module"])
            run(
                "stdin",
                ["-"],
                "from pathlib import Path\nassert Path.cwd().name == 'training project'\nprint('STDIN_PASS')\n",
            )
            run("exit-status", ["-c", "raise SystemExit(17)"], expected=17)
            run(
                "shell",
                [
                    "--shell",
                    "-c",
                    "test -f 'checkpoints with spaces/model checkpoint.pt' && echo SHELL_FILE_PASS",
                ],
            )
            run(
                "file-rw-mount",
                [
                    "--mount",
                    str(data),
                    "-c",
                    f"from pathlib import Path; p=Path({str(data)!r}); p.write_text('container update'); assert p.read_text()=='container update'",
                ],
            )
            assert data.read_text() == "container update"
            run("missing-file", ["/does-not-exist/train.py"], expected=2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    report = {
        "status": "passed",
        "release": str(release),
        "checks": checks,
        "checkpoint": str(checkpoint),
        "checkpoint_uid": checkpoint.stat().st_uid,
        "checkpoint_gid": checkpoint.stat().st_gid,
    }
    (logs / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LAUNCHER_HOST_IO_VALIDATION_PASS", flush=True)


if __name__ == "__main__":
    main()
