# SPDX-License-Identifier: MIT
"""Regression checks for retained Ninja dependency accounting."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from audit_link_headers_d54 import parse_header_records


class RetainedDependencyTest(unittest.TestCase):
    def test_counts_stale_outputs_without_dropping_their_headers(self):
        paths, records = parse_header_records(
            "lib/a.o: #deps 2, deps mtime 43 (VALID)\n"
            "    /sys/include/a.h\n    path with spaces.h\n\n"
            "lib/b.o: #deps 2, deps mtime 44 (STALE)\n"
            "    /sys/include/a.h\n    generated.h\n\n"
        )
        self.assertEqual(records, [("lib/a.o", "VALID"), ("lib/b.o", "STALE")])
        self.assertEqual(
            paths, {"/sys/include/a.h", "path with spaces.h", "generated.h"}
        )

    def test_rejects_missing_or_truncated_dependency_records(self):
        for data in ("", "a.o: #deps 2, deps mtime 42 (VALID)\n    a.h\n"):
            with self.subTest(data=data), self.assertRaises(AssertionError):
                parse_header_records(data)
