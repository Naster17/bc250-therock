# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check native scalar dot lowering and the unaffected packed RDNA path."""

import argparse
import json
from pathlib import Path
import re
import subprocess


def dot_module(architecture: str, operand_type: str) -> str:
    result_type = "i32" if operand_type == "i8" else "f32"
    a = f"tensor<16x16x{operand_type}, #ttg.dot_op<{{opIdx = 0, parent = #blocked}}>>"
    b = f"tensor<16x8x{operand_type}, #ttg.dot_op<{{opIdx = 1, parent = #blocked}}>>"
    c = f"tensor<16x8x{result_type}, #blocked>"
    return f"""#blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:{architecture}", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func private @dot_{operand_type}(%a: {a}, %b: {b}, %c: {c}) -> {c} {{
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : {a} * {b} -> {c}
    tt.return %d : {c}
  }}
}}
"""


def multiply_module(architecture: str) -> str:
    tensor = "tensor<128xbf16, #blocked>"
    return f"""#blocked = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:{architecture}", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func private @multiply_bf16(%a: {tensor}, %b: {tensor}) -> {tensor} {{
    %c = arith.mulf %a, %b : {tensor}
    tt.return %c : {tensor}
  }}
}}
"""


def validate(triton_opt: Path, output_dir: Path, llvm_bin: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").unlink(missing_ok=True)
    results = []
    for architecture in ("gfx1013", "gfx1100"):
        for operand_type in ("f16", "bf16", "i8", "bf16-multiply"):
            stem = f"{architecture}-{operand_type}"
            source = output_dir / f"{stem}.mlir"
            source.write_text(
                multiply_module(architecture)
                if operand_type == "bf16-multiply"
                else dot_module(architecture, operand_type)
            )
            completed = subprocess.run(
                [
                    str(triton_opt),
                    str(source),
                    "--allocate-shared-memory",
                    f"--convert-triton-amdgpu-to-llvm=gfx-arch={architecture}",
                    "--convert-builtin-func-to-llvm",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            lowered = completed.stdout
            (output_dir / f"{stem}.llvm.mlir").write_text(lowered)
            if architecture == "gfx1013":
                assert not re.search(r"amdgcn\.(?:[fsu]dot|mfma|wmma)", lowered), stem
                if operand_type == "bf16-multiply":
                    expected = {"llvm.fmul": 1, "llvm.zext": 2, "llvm.shl": 2}
                    assert "bf16 to i16" in lowered, stem
                elif operand_type == "i8":
                    expected = {"llvm.sext": 32, "llvm.mul": 16, "llvm.add": 16}
                    assert "i8 to i32" in lowered, stem
                    assert "llvm.zext" not in lowered, stem
                else:
                    expected = {'llvm.call_intrinsic "llvm.fmuladd.f32"': 16}
                    if operand_type == "f16":
                        expected["llvm.fpext"] = 32
                        assert "f16 to f32" in lowered, stem
                    else:
                        expected.update({"llvm.zext": 32, "llvm.shl": 32})
                        assert "bf16 to i16" in lowered, stem
                        assert "i32 to f32" in lowered, stem
            else:
                intrinsic = {
                    "f16": "llvm.amdgcn.fdot2",
                    "bf16": "llvm.amdgcn.fdot2.f32.bf16",
                    "i8": "llvm.amdgcn.sdot4",
                    "bf16-multiply": "llvm.amdgcn.fdot2.bf16.bf16",
                }[operand_type]
                expected = {
                    f'llvm.call_intrinsic "{intrinsic}"': (
                        1
                        if operand_type == "bf16-multiply"
                        else (4 if operand_type == "i8" else 8)
                    )
                }
                assert '"llvm.fmuladd.f32"' not in lowered, stem
            counts = {}
            for operation, count in expected.items():
                actual = len(re.findall(re.escape(operation) + r"\b", lowered))
                # Intrinsic names end in a quote, so word boundaries do not apply.
                if operation.endswith('"'):
                    actual = lowered.count(operation)
                assert actual == count, (stem, operation, actual, count)
                counts[operation] = actual
            llvm_ir = subprocess.run(
                [str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir"],
                input=lowered,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            # Export the fixture function so LLVM cannot discard an unreferenced
            # internal function before instruction selection.
            llvm_ir = llvm_ir.replace("define internal ", "define ")
            (output_dir / f"{stem}.ll").write_text(llvm_ir)
            assembly = subprocess.run(
                [
                    str(llvm_bin / "llc"),
                    "-mtriple=amdgcn-amd-amdhsa",
                    f"-mcpu={architecture}",
                    "-o",
                    "-",
                ],
                input=llvm_ir,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            ).stdout
            (output_dir / f"{stem}.s").write_text(assembly)
            function = (
                "multiply_bf16"
                if operand_type == "bf16-multiply"
                else f"dot_{operand_type}"
            )
            assert re.search(rf"^{function}:", assembly, re.MULTILINE), stem
            if architecture == "gfx1013":
                assert not re.search(r"\bv_(?:dot[24]|mfma|wmma)", assembly), stem
            results.append({"case": stem, "operations": counts})
            print(f"TRITON_DOT_LOWERING_PASS {stem}", flush=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-opt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    args = parser.parse_args()
    validate(
        args.triton_opt.resolve(), args.output_dir.resolve(), args.llvm_bin.resolve()
    )


if __name__ == "__main__":
    main()
