# SPDX-License-Identifier: MIT
"""Streaming must reject incomplete/duplicate callbacks and insufficient disk reserve."""
import importlib.util
from pathlib import Path
import unittest

BASE = Path(__file__).parents[1] / "bc250"
spec = importlib.util.spec_from_file_location(
    "stream_contract_d37", BASE / "stream_contract_d37.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StreamContractTest(unittest.TestCase):
    def test_complete_unique_names(self):
        tracker = module.Completion(["embedding", "block"])
        tracker.accept("block")
        tracker.accept("embedding")
        tracker.finish()

    def test_unknown_and_duplicate_callbacks_refuse(self):
        tracker = module.Completion(["embedding"])
        with self.assertRaises(ValueError):
            tracker.accept("head-alias")
        tracker.accept("embedding")
        with self.assertRaises(ValueError):
            tracker.accept("embedding")
        for names in [[], ["embedding", "embedding"]]:
            with self.assertRaises(ValueError):
                module.Completion(names)

    def test_missing_callback_refuses(self):
        tracker = module.Completion(["embedding", "block"])
        tracker.accept("block")
        with self.assertRaises(ValueError):
            tracker.finish()

    def test_one_tensor_disk_bound(self):
        scratch = 100663296
        required = 5 * module.GIB + scratch
        with self.assertRaises(ValueError):
            module.admit_scratch(required - 1, scratch)
        self.assertEqual(
            module.admit_scratch(required, scratch)["maximum_scratch_bytes"], scratch
        )
        for bound in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                module.admit_scratch(required, bound)
