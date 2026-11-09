# SPDX-License-Identifier: MIT
import sys, struct, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from finite_reference_d41 import check_chunk


class FiniteTest(unittest.TestCase):
    def test_finite_boundaries_both_signs_and_widths(self):
        for width, kind, largest in [
            (4, "f", 3.4028234663852886e38),
            (8, "d", sys.float_info.max),
        ]:
            check_chunk(
                struct.pack("<" + kind * 6, 0.0, -0.0, 1.0, -1.0, largest, -largest),
                width,
            )

    def test_nonfinite_any_position_both_widths(self):
        for width, kind in [(4, "f"), (8, "d")]:
            for value in [float("inf"), float("-inf"), float("nan")]:
                for position in range(4):
                    values = [0.0, 1.0, -1.0, 0.0]
                    values[position] = value
                    with self.assertRaises(ValueError):
                        check_chunk(struct.pack("<" + kind * 4, *values), width)

    def test_invalid_word_alignment_refused(self):
        with self.assertRaises(ValueError):
            check_chunk(b"\x00", 4)
        with self.assertRaises(ValueError):
            check_chunk(b"\x00" * 8, 16)
