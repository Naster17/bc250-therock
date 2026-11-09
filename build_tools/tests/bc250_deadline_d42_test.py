# SPDX-License-Identifier: MIT
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/deadline_guard_d42.py"


class DeadlineTest(unittest.TestCase):
    def run_guard(self, seconds):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name) / "guard"
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--release",
                os.uname().release,
                "--timeout",
                str(seconds),
                "--directory",
                str(directory),
                "--dry-run",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(lambda: process.poll() is None and process.kill())
        return process, directory

    def test_extended_bound_acknowledges_exact_nonce(self):
        process, directory = self.run_guard(1800)
        for _ in range(100):
            if (directory / "state.json").exists():
                break
            time.sleep(0.01)
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["status"], "armed")
        self.assertEqual(state["timeout_seconds"], 1800)
        (directory / "ack.json").write_text(
            json.dumps({k: state[k] for k in ["nonce", "boot_id", "phase"]})
        )
        process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(
            json.loads((directory / "state.json").read_text())["status"], "acknowledged"
        )

    def test_short_dry_expiry_cannot_reboot(self):
        process, directory = self.run_guard(1)
        process.communicate(timeout=3)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(
            json.loads((directory / "state.json").read_text())["status"], "expired"
        )

    def test_above_reviewed_maximum_refuses(self):
        process, directory = self.run_guard(1801)
        process.communicate(timeout=3)
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse((directory / "state.json").exists())


if __name__ == "__main__":
    unittest.main()
