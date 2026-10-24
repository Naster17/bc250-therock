#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute unchanged architecture predicates; this does not qualify BF16 graphs."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (args.source / "src/targets/gpu/device_name.cpp").read_text()
    names = [
        "gfx_has_fp8fnuz_intrinsics_impl",
        "gfx_has_fp8ocp_intrinsics_impl",
        "gfx_has_bf16_intrinsics_impl",
        "gfx_has_mx_intrinsics_impl",
        "hipblaslt_supported_impl",
    ]
    functions = []
    for name in names:
        matches = re.findall(r"static bool " + name + r"\([^}]+\n}", source)
        assert len(matches) == 1, name
        functions += matches
    code = "#include <string>\n#include <iostream>\n"
    code += "bool starts_with(const std::string& s, const std::string& p) { return s.rfind(p,0)==0; }\n"
    code += "\n".join(functions)
    code += "\nint main() {\n"
    for arch in ["gfx1013", "gfx1030", "gfx1100", "gfx942", "gfx950", "gfx1200"]:
        for name in names:
            code += f'std::cout << "{arch} {name} " << {name}("{arch}") << "\\n";\n'
    code += "}\n"
    cpp = args.scratch / "feature-admission.cpp"
    cpp.write_text(code)
    exe = args.scratch / "feature-admission"
    subprocess.run(
        ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)],
        check=True,
    )
    output = subprocess.check_output([str(exe)], text=True)
    values = {
        arch: {}
        for arch in ["gfx1013", "gfx1030", "gfx1100", "gfx942", "gfx950", "gfx1200"]
    }
    for line in output.splitlines():
        arch, name, value = line.split()
        values[arch][name] = bool(int(value))
    assert values["gfx1013"] == dict(zip(names, [False, False, True, False, False]))
    assert not values["gfx1030"]["gfx_has_bf16_intrinsics_impl"]
    assert values["gfx942"]["gfx_has_fp8fnuz_intrinsics_impl"]
    assert values["gfx950"]["gfx_has_mx_intrinsics_impl"]
    assert values["gfx1200"]["gfx_has_fp8ocp_intrinsics_impl"]
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                values=values,
                source_functions=functions,
                gpu_graph_qualification=False,
                bf16="Existing predicate true; dedicated graph validation still required",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
