#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify installed package identities; reuse unchanged independent W8 reference suites."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group",
        choices=("imports", "module", "state", "resume", "recipe"),
        required=True,
    )
    args = parser.parse_args()
    import torch

    linear = torch.nn.Linear
    native = torch._weight_int8pack_mm
    import bc250_quant as public
    from bc250_quant import _w8, modules

    root = Path(public.__file__).resolve().parent
    expected = Path("/opt/torch-venv/lib/python3.12/site-packages/bc250_quant")
    assert root == expected
    assert modules.FrozenW8Linear is public.FrozenW8Linear
    assert modules.quant is _w8
    assert public.pack is _w8.pack and public.linear is _w8.linear
    assert torch.nn.Linear is linear and torch._weight_int8pack_mm is native
    assert (
        "bc250_quant_b051" not in sys.modules and "bc250_quant_b055" not in sys.modules
    )
    if args.group == "imports":
        report = dict(status="running", checks=[])

        def passed(name):
            report["checks"].append(name)

        try:
            distribution = importlib.metadata.distribution("bc250-quant")
            assert distribution.version == public.__version__ == "0.1.0"
            assert distribution.requires == ["torch"]
            passed("installed-public-path-version-exports-no-dispatch-hooks")
            for item in distribution.files:
                if item.hash:
                    data = Path(distribution.locate_file(item)).read_bytes()
                    assert item.hash.mode == "sha256"
                    assert (
                        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                        .rstrip(b"=")
                        .decode()
                        == item.hash.value
                    )
                    assert len(data) == item.size
            passed("installed-wheel-record-hashes")
            source = Path(__file__).resolve().parent
            provenance = json.loads((root / "provenance.json").read_text())
            for name, digest in provenance["source_sha256"].items():
                original = (source / name).read_bytes()
                assert hashlib.sha256(original).hexdigest() == digest
                packaged = (
                    root / ("_w8.py" if name.endswith("b051.py") else "modules.py")
                ).read_bytes()
                if name.endswith("b055.py"):
                    packaged = packaged.replace(
                        b"from . import _w8 as quant",
                        b"import bc250_quant_b051 as quant",
                    )
                assert original == packaged
            passed("exact-qualified-core-module-except-relative-import")
            assert torch.cuda.get_device_properties(0).gcnArchName.startswith(
                "gfx1013:"
            )
            import bc250_fft
            import bc250_sparse
            import torchvision
            import torchaudio
            import torchcodec

            assert callable(bc250_fft.fft) and callable(bc250_sparse.spsolve)
            assert all(m.__version__ for m in (torchvision, torchaudio, torchcodec))
            passed("native-device-and-retained-optional-imports")
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    # Reference scripts stay byte-identical. Only bind their helper import to the
    # installed public implementation, never a helper-directory implementation.
    sys.modules["bc250_quant_b051"] = _w8
    sys.modules["bc250_quant_b055"] = public
    script = (
        "validate_quant_b051.py" if args.group == "recipe" else "validate_quant_b055.py"
    )
    sys.argv = [script, "--output", str(args.output)]
    if args.group != "recipe":
        sys.argv += ["--group", args.group]
    runpy.run_path(str(Path(__file__).resolve().parent / script), run_name="__main__")


if __name__ == "__main__":
    main()
