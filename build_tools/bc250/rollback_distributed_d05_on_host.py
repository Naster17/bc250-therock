#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Restore the retained D01 distributed tool and registry under the host EX lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path


def main():
    assert os.geteuid() == 0 and os.uname().nodename in ["bc250", "bc500"]
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    current = json.loads(registry.read_text())
    record = Path("/home/nik/bc250work/evidence/distributed-d05-tool-install-r1")
    activation = json.loads((record / "activation.json").read_text())
    command = Path("/usr/local/bin/bc250-distributed")
    with Path(current["lock"]).open("r") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.readlink(command) == activation["new_command"]
        before = json.loads((record / "registry-before.json").read_text())
        runtime = Path(before["release"]) / "runtime.json"
        assert (
            hashlib.sha256(runtime.read_bytes()).hexdigest()
            == before["runtime_sha256"]
            == activation["runtime_sha256"]
        )
        assert Path(activation["old_command"]).is_file()
        temp = registry.with_suffix(".d05-rollback")
        temp.write_bytes((record / "registry-before.json").read_bytes())
        temp.chmod(0o444)
        temp.replace(registry)
        temp = command.with_name("bc250-distributed-d05-rollback")
        temp.symlink_to(activation["old_command"])
        temp.replace(command)
        (record / "rollback.json").write_text(
            json.dumps(
                dict(
                    status="passed",
                    restored_command=activation["old_command"],
                    native_runtime_unchanged=True,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
