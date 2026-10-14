# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import fcntl
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/bc250_torch.py"
spec = importlib.util.spec_from_file_location("bc250_runtime", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class RuntimeLockTest(unittest.TestCase):
    def test_shared_workload_excludes_update_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.lock"
            with self.assertRaisesRegex(RuntimeError, "workload failed"):
                with launcher.runtime_lock(path):
                    with path.open("r+") as peer:
                        fcntl.flock(peer, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        fcntl.flock(peer, fcntl.LOCK_UN)
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(peer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise RuntimeError("workload failed")
            with path.open("r+") as peer:
                fcntl.flock(peer, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_update_excludes_workloads_and_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.lock"
            with launcher.runtime_lock(path, exclusive=True):
                with path.open("r+") as peer:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(peer, fcntl.LOCK_SH | fcntl.LOCK_NB)
            link = Path(directory) / "linked.lock"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                with launcher.runtime_lock(link):
                    self.fail("accepted symlink")


if __name__ == "__main__":
    unittest.main()
