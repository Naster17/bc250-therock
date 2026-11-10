# SPDX-License-Identifier: MIT
"""Fail closed on backend drift and retained build-version ambiguity."""
import importlib.util
from pathlib import Path
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "bc250"
spec = importlib.util.spec_from_file_location(
    "inputs_d51", TOOLS / "build_inputs_d51.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BuildInputsTest(unittest.TestCase):
    def setUp(self):
        self.cache = "\n".join(k + ":BOOL=0" for k in m.OFF_FLAGS)
        self.missing = [
            dict(path="third_party/mslk/external/dep" + str(i), pin=str(i) * 40)
            for i in range(8)
        ]

    def test_disabled_native_configuration(self):
        result = m.classify(
            self.cache,
            [dict(file="native.cpp", command="g++ native.cpp")],
            self.missing,
        )
        self.assertEqual(result["compile_entries"], 1)
        self.assertEqual(len(result["missing"]), 8)

    def test_flag_drift_or_missing_flag_refused(self):
        for value in ["1", "ON", "garbage"]:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                m.classify(
                    self.cache.replace("USE_MSLK:BOOL=0", "USE_MSLK:BOOL=" + value),
                    [],
                    self.missing,
                )
        with self.assertRaises(AssertionError):
            m.classify("", [], self.missing)

    def test_include_reference_refused(self):
        with self.assertRaises(AssertionError):
            m.classify(
                self.cache,
                [
                    dict(
                        file="native.cpp",
                        command="g++ -Ithird_party/aiter/include native.cpp",
                    )
                ],
                self.missing,
            )

    def test_retained_build_scripts_parse_without_execution(self):
        for name in ["b052", "b053"]:
            result = m.build_settings(
                (TOOLS / ("build_torch_" + name + ".sh")).read_text()
            )
            self.assertEqual(result["PYTORCH_BUILD_VERSION"], "2.13.0a0+gfx1013")
            self.assertEqual(result["USE_NCCL"], "0" if name == "b052" else "1")
        with self.assertRaises(AssertionError):
            m.build_settings("export PYTORCH_BUILD_VERSION=changed")


if __name__ == "__main__":
    unittest.main()
