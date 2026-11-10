# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import struct
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/audit_elf_d46.py"
spec = importlib.util.spec_from_file_location("elf_d46", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(path):
    data = bytearray(512)
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        data,
        0,
        b"\x7fELF\x02\x01" + b"\0" * 10,
        3,
        62,
        1,
        0,
        0,
        192,
        0,
        64,
        0,
        0,
        64,
        5,
        4,
    )
    data[64:68] = b"code"
    data[68:100] = b"\0" + path.encode() + b"\0" * (31 - len(path))
    struct.pack_into("<qQqQ", data, 100, 29, 1, 0, 0)
    names = b"\0.text\0.dynstr\0.dynamic\0.shstrtab\0"
    data[132 : 132 + len(names)] = names
    for index, values in enumerate(
        [
            (0, 0, 0, 0, 0),
            (1, 1, 6, 64, 4),
            (7, 3, 2, 68, 32),
            (15, 6, 2, 100, 32),
            (24, 3, 0, 132, len(names)),
        ]
    ):
        name, kind, flags, offset, size = values
        struct.pack_into(
            "<IIQQQQIIQQ",
            data,
            192 + 64 * index,
            name,
            kind,
            flags,
            0,
            offset,
            size,
            0,
            0,
            1,
            0,
        )
    return bytes(data)


class ElfAuditTest(unittest.TestCase):
    def test_exact_runtime_path_transformation(self):
        result = audit.compare(fixture("/long/build/lib:"), fixture("$ORIGIN"))
        self.assertTrue(result["rpath_only_exact_replay"])
        self.assertEqual(result["changed_sections"], [".dynstr"])

    def test_code_change_is_not_packaging_only(self):
        changed = bytearray(fixture("$ORIGIN"))
        changed[64] = 0
        self.assertFalse(
            audit.compare(fixture("/long/build/lib:"), changed)[
                "rpath_only_exact_replay"
            ]
        )

    def test_invalid_headers_and_truncation(self):
        for data in (b"not ELF", fixture("$ORIGIN")[:200]):
            with self.assertRaises(ValueError):
                audit.elf(data)

    def test_invalid_dynamic_string_offset(self):
        data = bytearray(fixture("$ORIGIN"))
        struct.pack_into("<Q", data, 108, 32)
        with self.assertRaises(ValueError):
            audit.elf(data)


if __name__ == "__main__":
    unittest.main()
