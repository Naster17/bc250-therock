#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Serve a bounded build-storage SSHFS slave over an existing SSH connection.

The root-owned sshfs process and mode-0600 FIFOs must already be prepared on
.104. No listening network service, credential file or SSH configuration change
is needed. The mount is temporary; its backing artifacts remain on local disk.
"""
import argparse
from pathlib import Path
import signal
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--host", default="nik@192.168.1.104")
    parser.add_argument(
        "--fifo-root",
        default="/home/nik/bc250work/rocm-10.2-gfx1013/tmp/migraphx-b064/20261025-storage",
    )
    args = parser.parse_args()
    fifo = Path(args.fifo_root)
    assert (
        fifo.is_absolute()
        and str(fifo)
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
        .isalnum()
    )
    args.log_directory.mkdir(parents=True, exist_ok=True)
    with (args.log_directory / "bridge.log").open("ab") as log:
        ssh = subprocess.Popen(
            [
                "ssh",
                "-F",
                "/dev/null",
                "-T",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                args.host,
                f"cat {fifo}/output & cat > {fifo}/input; wait",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        server = subprocess.Popen(
            ["/usr/lib/ssh/sftp-server"], stdin=ssh.stdout, stdout=ssh.stdin, stderr=log
        )
        ssh.stdin.close()
        ssh.stdout.close()

        def stop(*_):
            server.terminate()
            ssh.terminate()

        signal.signal(signal.SIGTERM, stop)
        print(f"BRIDGE ssh={ssh.pid} sftp={server.pid}", flush=True)
        try:
            code = server.wait()
            print("SERVER_EXIT", code, flush=True)
        finally:
            ssh.terminate()
            ssh.wait(timeout=10)


if __name__ == "__main__":
    main()
