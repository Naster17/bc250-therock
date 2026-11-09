# SPDX-License-Identifier: MIT
"""The per-unit audit must never omit parameters or confuse prefix boundaries."""
import importlib.util
from pathlib import Path
import unittest

BASE = Path(__file__).parents[1] / "bc250"
spec = importlib.util.spec_from_file_location(
    "unit_plan_d36", BASE / "unit_plan_d36.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class UnitPlanTest(unittest.TestCase):
    def test_deepest_owner_and_prefix_boundary(self):
        names = [
            "embedding.weight",
            "blocks.1.weight",
            "blocks.10.weight",
            "blocks.1.inner.weight",
            "blocks.1extra.weight",
        ]
        result = module.partition(
            names, ["", "blocks.1", "blocks.10", "blocks.1.inner"]
        )
        self.assertEqual(result[""], ["embedding.weight", "blocks.1extra.weight"])
        self.assertEqual(result["blocks.1"], ["blocks.1.weight"])
        self.assertEqual(result["blocks.10"], ["blocks.10.weight"])
        self.assertEqual(result["blocks.1.inner"], ["blocks.1.inner.weight"])
        self.assertEqual(sum(map(len, result.values())), len(names))

    def test_missing_root_duplicate_and_empty_units_refuse(self):
        for names, prefixes in [
            (["x"], ["x"]),
            (["x", "x"], [""]),
            (["x"], ["", ""]),
            (["x"], ["", "missing"]),
            ([], [""]),
        ]:
            with self.assertRaises(ValueError):
                module.partition(names, prefixes)

    def test_tied_gpt2_named_coverage(self):
        names = [
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "transformer.ln_f.weight",
            "transformer.ln_f.bias",
        ]
        for block in range(12):
            names.extend(
                f"transformer.h.{block}.p{parameter}" for parameter in range(12)
            )
        prefixes = [""] + [f"transformer.h.{i}" for i in range(12)]
        result = module.partition(names, prefixes)
        self.assertEqual(len(names), 148)
        self.assertEqual(len(result[""]), 4)
        self.assertTrue(all(len(result[p]) == 12 for p in prefixes[1:]))
        self.assertEqual({n for group in result.values() for n in group}, set(names))
