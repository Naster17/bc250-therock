#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Public-name ownership plan; refuses empty, duplicate or ambiguous audit units."""


def partition(names, prefixes):
    names, prefixes = list(names), list(prefixes)
    if len(names) != len(set(names)) or len(prefixes) != len(set(prefixes)):
        raise ValueError("Duplicate audit parameter/unit")
    if "" not in prefixes or not names:
        raise ValueError("Missing root/parameter coverage")
    result = {prefix: [] for prefix in prefixes}
    for name in names:
        if not isinstance(name, str) or not name or name.startswith("."):
            raise ValueError("Invalid parameter name")
        candidates = [
            prefix for prefix in prefixes if not prefix or name.startswith(prefix + ".")
        ]
        owner = max(candidates, key=len)
        result[owner].append(name)
    if any(not group for group in result.values()):
        raise ValueError("Empty audit unit")
    assert sum(map(len, result.values())) == len(names)
    return result
