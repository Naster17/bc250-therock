#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""EX-locked versioned install of the qualified distributed consumer environment launcher."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--payload-root", type=Path)
    args = parser.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    data = json.loads(registry.read_text())
    source = args.payload_root or Path(__file__).resolve().parent / "install-payload"
    pins = json.loads((source / "payload-hashes.json").read_text())
    expected = {
        "bc250_distributed.py",
        "bc250_torch.py",
        "bc250_torchrun.py",
        "bc250_distributed_identity.py",
        "bc250_distributed_worker.py",
        "torch-identity.json",
        "agent-api-source.py",
        "torchrun-fix.json",
        "pytorch-LICENSE",
    }
    assert set(pins) == expected, "Unexpected installer payload names"
    assert all(sha(source / name) == value for name, value in pins.items())
    fix = json.loads((source / "torchrun-fix.json").read_text())
    base = Path(data["release"])
    assert sha(base / "runtime.json") == data["runtime_sha256"]
    assert (
        sha(base / "torch/torch/distributed/elastic/agent/server/api.py")
        == fix["original_agent_api_sha256"]
    )
    with Path(data["lock"]).open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            json.loads(registry.read_text()) == data
        ), "Registry changed before lock acquisition"
        assert sha(base / "runtime.json") == data["runtime_sha256"]
        folder = Path("/opt/bc250-stack/tools/distributed-d09")
        record = Path("/home/nik/bc250work/evidence/distributed-d09-tool-install-r1")
        command = Path("/usr/local/bin/bc250-distributed")
        old_command = os.readlink(command)
        if not args.activate:
            assert not folder.exists()
            record.mkdir()
            (record / "registry-before.json").write_bytes(registry.read_bytes())
            folder.mkdir()
            for name in pins:
                shutil.copyfile(source / name, folder / name)
                (folder / name).chmod(0o555 if name.endswith(".py") else 0o444)
            (record / "old-command.json").write_text(
                json.dumps(dict(old_command=old_command))
            )
            print("Staged versioned candidate; active command and registry unchanged")
            return
        assert all(sha(folder / name) == value for name, value in pins.items())
        assert (
            old_command
            == json.loads((record / "old-command.json").read_text())["old_command"]
        )
        assert registry.read_bytes() == (record / "registry-before.json").read_bytes()
        for label in ["consumer-candidate", "retained-candidate"]:
            report = (
                Path("/home/nik/bc250work/evidence/distributed-d09-r1")
                / label
                / "final.json"
            )
            qualification = json.loads(report.read_text())
            assert (
                qualification["status"] == "passed"
                and qualification["agent_shutdown_clean"]
            )
            assert qualification["selected_tools"] == str(folder)
            assert qualification["launcher_sha256"] == pins["bc250_distributed.py"]
            assert (
                qualification["kernel_log_unchanged"]
                and qualification["registry_unchanged"]
            )
        data["launcher_tools"] = str(folder)
        data["agent_exit_fix"] = fix
        data["consumer_environment"] = dict(
            schema=1,
            recovery=str(record),
            torch_identity_sha256=pins["torch-identity.json"],
            launcher_sha256=pins["bc250_distributed.py"],
        )
        temp = registry.with_suffix(".d09-new")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.chmod(0o444)
        temp.replace(registry)
        (record / "activation.json").write_text(
            json.dumps(
                dict(
                    old_command=old_command,
                    new_command=str(folder / "bc250_distributed.py"),
                    runtime_sha256=data["runtime_sha256"],
                    payload_hashes=pins,
                ),
                indent=2,
            )
            + "\n"
        )
        temp = command.with_name("bc250-distributed-d09-new")
        temp.symlink_to(folder / "bc250_distributed.py")
        temp.replace(command)
        assert sha(base / "runtime.json") == data["runtime_sha256"]
        print(
            json.dumps(
                dict(
                    status="activated-after-candidate-qualification",
                    tool=str(folder),
                    old_tool=old_command,
                )
            )
        )


if __name__ == "__main__":
    main()
