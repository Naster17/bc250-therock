#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create the additive L02 entry point from the qualified L01 script."""
import argparse
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
data = args.source.read_bytes()
assert (
    hashlib.sha256(data).hexdigest()
    == "832017cde0782f153e8f06920dfc548234e8235a13ac0265dbc3c0e863332f22"
)
text = data.decode()
for old in [
    "from bc250_lora_support import configure, Audit",
    "from bc250_lora_support import configure_trainer",
]:
    assert text.count(old) == 1
    text = text.replace(old, old.replace("bc250_lora_support", "bc250_lora_checkpoint"))
if args.destination.exists():
    assert args.destination.read_text() == text, "Existing entry point differs"
else:
    args.destination.write_text(text)
