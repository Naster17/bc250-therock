# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from concurrent.futures import ThreadPoolExecutor
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "linux", "ROCTX wrapper is Linux-only")
class RoctxTest(unittest.TestCase):
    def test_upstream_marker_abi_and_thread_local_ranges(self):
        upstream = ROOT / "rocm-systems/projects/roctracer"
        if not (upstream / "src/roctx/roctx.cpp").is_file():
            self.skipTest("ROCtracer submodule is not initialized")
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            stage = Path(directory) / "stage"
            commands = [
                [
                    "cmake",
                    "-S",
                    str(ROOT / "profiler/roctx"),
                    "-B",
                    str(build),
                    f"-DROCTRACER_SOURCE_DIR={upstream}",
                    f"-DCMAKE_INSTALL_PREFIX={stage}",
                    "-DCMAKE_INSTALL_LIBDIR=lib",
                ],
                ["cmake", "--build", str(build), "--parallel", "2"],
                ["cmake", "--install", str(build)],
            ]
            for command in commands:
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((stage / "include/roctracer/roctx.h").is_file())
            library = stage / "lib/libroctx64.so.4"
            symbols = subprocess.check_output(
                ["readelf", "--dyn-syms", "--wide", str(library)], text=True
            )
            self.assertIn("roctxMarkA@@ROCTX_4.1", symbols)
            roctx = ctypes.CDLL(str(library))
            roctx.roctxRangeStartA.restype = ctypes.c_uint64
            roctx.roctxRangeStop.argtypes = [ctypes.c_uint64]
            self.assertEqual(roctx.roctx_version_major(), 4)
            self.assertEqual(roctx.roctx_version_minor(), 1)
            self.assertEqual(roctx.roctxRangePop(), -1)

            def nested_ranges(_: int) -> list[int]:
                return [
                    roctx.roctxRangePushA(b"outer"),
                    roctx.roctxRangePushA(b"inner"),
                    roctx.roctxRangePop(),
                    roctx.roctxRangePop(),
                    roctx.roctxRangePop(),
                ]

            with ThreadPoolExecutor(max_workers=4) as workers:
                for result in workers.map(nested_ranges, range(16)):
                    self.assertEqual(result, [0, 1, 1, 0, -1])
            first = roctx.roctxRangeStartA(b"first")
            second = roctx.roctxRangeStartA(b"second")
            self.assertGreater(first, 0)
            self.assertNotEqual(first, second)
            roctx.roctxRangeStop(first)
            roctx.roctxRangeStop(second)
            roctx.roctxMarkA(b"marker")


if __name__ == "__main__":
    unittest.main()
