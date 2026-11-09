# SPDX-License-Identifier: MIT
"""Reject drift of the native checkpoint recipe before an oracle can be trusted."""
from functools import partial
import importlib.util
from pathlib import Path
import unittest

BASE = Path(__file__).parents[1] / "bc250"
spec = importlib.util.spec_from_file_location(
    "cpu_checkpoint_policy_d38", BASE / "cpu_checkpoint_policy_d38.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CPUCheckpointPolicyTest(unittest.TestCase):
    def test_explicit_native_recipe(self):
        module.validate_options(dict(module.OPTIONS))
        native = lambda: None
        self.assertTrue(
            module.callable_matches(partial(native, **module.OPTIONS), native)
        )

    def test_missing_reentrant_rng_and_integer_drift_refuse(self):
        for options in [
            {},
            {"use_reentrant": True, "preserve_rng_state": True},
            {"use_reentrant": False, "preserve_rng_state": False},
            {"use_reentrant": 0, "preserve_rng_state": 1},
            {**module.OPTIONS, "debug": True},
        ]:
            with self.assertRaises(ValueError):
                module.validate_options(options)

    def test_wrong_callable_or_positional_arguments_refuse(self):
        native = lambda: None
        other = lambda: None
        self.assertFalse(module.callable_matches(native, native))
        self.assertFalse(
            module.callable_matches(partial(other, **module.OPTIONS), native)
        )
        self.assertFalse(
            module.callable_matches(partial(native, 1, **module.OPTIONS), native)
        )
        self.assertFalse(
            module.callable_matches(
                partial(native, use_reentrant=True, preserve_rng_state=True), native
            )
        )
