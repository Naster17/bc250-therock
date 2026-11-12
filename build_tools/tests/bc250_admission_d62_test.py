# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/admit_model_d62.py"
spec = importlib.util.spec_from_file_location("admit_model_d62", SCRIPT)
admit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admit)


class OwnedImportGuardTest(unittest.TestCase):
    def test_clean_path_has_no_violations(self):
        self.assertEqual(
            admit.owned_import_guard(["/usr/lib/python3", "/opt/site"]), []
        )

    def test_excluded_application_path_is_refused(self):
        path = [
            "/usr/lib/python3",
            "/home/nik/finetune-n7/runtime/transformers-5.19.0-cp312",
        ]
        self.assertEqual(
            admit.owned_import_guard(path),
            ["/home/nik/finetune-n7/runtime/transformers-5.19.0-cp312"],
        )

    def test_fixture_source_makes_no_forbidden_load_calls(self):
        source = SCRIPT.read_text()
        self.assertNotIn("snapshot_download(", source)
        self.assertNotIn(".from_pretrained(", source)


class RegistryTest(unittest.TestCase):
    def test_known_models_have_required_fields(self):
        for name in ("gpt2-tiny", "qwen3-tiny", "qwen35-hybrid-tiny"):
            spec = admit.registry_spec(name)
            for field in (
                "architecture",
                "config_class",
                "config",
                "backends",
                "routes",
            ):
                self.assertIn(field, spec)
            self.assertTrue(spec["config"])
            self.assertTrue(spec["backends"])

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            admit.registry_spec("not-a-model")

    def test_gpt2_tiny_matches_d26_owned_identity(self):
        spec = admit.registry_spec("gpt2-tiny")
        self.assertEqual(spec["param_count"], 21280)
        self.assertEqual(spec["config"]["vocab_size"], 97)
        self.assertEqual(spec["config"]["n_embd"], 32)

    def test_hybrid_requires_separate_overlay(self):
        spec = admit.registry_spec("qwen35-hybrid-tiny")
        self.assertEqual(spec["requires_transformers"], "5.19.0")
        self.assertIn("gated-delta", spec["routes"])
        self.assertIn("linear_attention", spec["config"]["layer_types"])

    def test_registry_routes_are_known(self):
        for spec in admit.MODEL_REGISTRY.values():
            for route in spec["routes"]:
                self.assertIn(route, admit.ROUTE_NAMES)


class RouteStatusTest(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(admit.route_status(True, False, False), "fired")
        self.assertEqual(admit.route_status(False, True, False), "refused")
        self.assertEqual(admit.route_status(False, False, True), "absent")
        self.assertEqual(admit.route_status(False, False, False), "not-exercised")

    def test_absent_takes_precedence(self):
        self.assertEqual(admit.route_status(True, True, True), "absent")


class VersionRecordTest(unittest.TestCase):
    def test_drops_empty_and_sorts(self):
        record = admit.version_record(
            {"trl": "", "torch": "2.13.0a0", "peft": "0.18.1"}
        )
        self.assertEqual(record, {"peft": "0.18.1", "torch": "2.13.0a0"})


class ReportTest(unittest.TestCase):
    def test_new_report_skeleton(self):
        report = admit.new_report("gpt2-tiny", "float32", "eager")
        self.assertEqual(report["fixture"], "admit_model_d62")
        self.assertEqual(report["status"], "running")
        self.assertEqual(report["checks"], [])
        self.assertEqual(report["routes"], {})

    def test_summarize_empty(self):
        self.assertEqual(admit.summarize(admit.new_report("m", "d", "b")), "empty")

    def test_summarize_passed_with_expected_refusals(self):
        report = admit.new_report("m", "d", "b")
        admit.add_check(report, "forward", "passed")
        admit.add_check(report, "tiled", "refused-expected", error="unsupported")
        admit.add_check(report, "hybrid", "absent")
        self.assertEqual(admit.summarize(report), "passed")

    def test_summarize_failed(self):
        report = admit.new_report("m", "d", "b")
        admit.add_check(report, "forward", "passed")
        admit.add_check(report, "backward", "failed")
        self.assertEqual(admit.summarize(report), "failed")

    def test_summarize_incomplete(self):
        report = admit.new_report("m", "d", "b")
        admit.add_check(report, "forward", "running")
        self.assertEqual(admit.summarize(report), "incomplete")

    def test_sha256_bytes(self):
        self.assertEqual(
            admit.sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )


if __name__ == "__main__":
    unittest.main()
