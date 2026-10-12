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
import signal
import subprocess
import tempfile
import threading
import time


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--launcher", type=Path, default=Path("/usr/local/bin/bc250-torch")
    )
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError("Run host orchestration as root")
    workspace = args.workspace.resolve()
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    uid, gid = manifest["uid"], manifest["gid"]
    root = Path(tempfile.mkdtemp(prefix="launcher-io-validation-", dir=workspace))
    project = root / "training project"
    project.mkdir()
    logs = workspace / "logs/launcher-io"
    logs.mkdir(exist_ok=True)
    launcher = str(args.launcher.resolve(strict=True))
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
        package_source = root / "fixture package"
        package_source.mkdir()
        (package_source / "setup.py").write_text(
            "from setuptools import setup\n"
            "setup(name='bc250-launcher-fixture', version='0.0.1', py_modules=['bc250_fixture'])\n"
        )
        (package_source / "bc250_fixture.py").write_text("VALUE = 1704\n")
        packages = root / "persistent packages"
        packages.mkdir()
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
            run(
                "package-install",
                [
                    "--env",
                    "PIP_NO_INDEX=1",
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(packages),
                    "--no-deps",
                    "--no-build-isolation",
                    str(package_source),
                ],
            )
            assert (packages / "bc250_fixture.py").stat().st_uid == uid
            run(
                "package-import-after-restart",
                [
                    "--env",
                    f"PYTHONPATH={packages}",
                    "-c",
                    "import bc250_fixture; assert bc250_fixture.VALUE == 1704; print('PERSISTENT_PACKAGE_PASS')",
                ],
            )
            run(
                "torch-compile",
                [
                    "-c",
                    (
                        "import torch; x=torch.arange(65539,dtype=torch.float32)/65539; "
                        "f=torch.compile(lambda x: torch.sin(x)+x*x,fullgraph=True); "
                        "torch.testing.assert_close(f(x.cuda()).cpu(),torch.sin(x)+x*x,rtol=1e-5,atol=1e-5); "
                        "print('LAUNCHER_COMPILE_PASS')"
                    ),
                ],
            )
            for signum, expected in ((signal.SIGTERM, 0), (signal.SIGINT, 130)):
                name = signal.Signals(signum).name.lower()
                ready = root / f"{name}-ready"
                saved = root / f"{name}-saved"
                code = (
                    "import signal,time; from pathlib import Path\n"
                    "def save(signum, frame):\n"
                    f"    Path({str(saved)!r}).write_text('saved before exit')\n"
                    "    raise SystemExit(0)\n"
                    f"signal.signal({int(signum)}, save)\n"
                    f"Path({str(ready)!r}).touch()\n"
                    "while True: time.sleep(0.1)\n"
                )
                with subprocess.Popen(
                    [launcher, "-c", code],
                    cwd=project,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ) as process:
                    deadline = time.monotonic() + 30
                    while not ready.exists() and process.poll() is None:
                        if time.monotonic() > deadline:
                            process.kill()
                            raise RuntimeError(f"Signal case {name} did not start")
                        time.sleep(0.1)
                    process.send_signal(signum)
                    output, _ = process.communicate(timeout=30)
                    (logs / f"{name}.log").write_text(output)
                    assert process.returncode == expected, (name, process.returncode)
                    assert saved.read_text() == "saved before exit"
                    checks.append(name)
                    print(f"LAUNCHER_CASE_PASS {name}", flush=True)
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
