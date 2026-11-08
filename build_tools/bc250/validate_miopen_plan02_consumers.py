#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual optional-package GPU consumers in the staged native precision cohort."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", choices=["ck", "migraphx"], required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(status="running", checks=[])
    try:
        import numpy as np
        import torch

        torch.set_num_threads(4)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        C.CDLL("libMIOpen.so.1")
        lib = Path("/workspace/build/dist/rocm/lib/libMIOpen.so.1.0")
        assert hashlib.sha256(lib.read_bytes()).hexdigest() == a.sha
        assert str(lib) in Path("/proc/self/maps").read_text()
        r["checks"].append("actual-normalized-native-MIOpen-identity")
        if a.route == "ck":
            sys.path.insert(0, "/home/nik/bc250work/rocm-10.2-gfx1013/helpers")
            from validate_ck_plugin_b078 import public_same_process

            r["maximum_CK_error"] = public_same_process()
            r["checks"].append("public137-FP32-CPU-reference-under-new-native-library")
        else:
            import migraphx as mx

            prog = mx.program()
            m = prog.get_main_module()
            x = m.add_parameter("x", mx.shape(type="float_type", lens=[2, 3]))
            y = m.add_instruction(mx.op("mul"), [x, x])
            m.add_return([y])
            prog.compile(mx.get_target("gpu"), offload_copy=True, fast_math=False)
            values = np.arange(6, dtype=np.float32).reshape(2, 3) / 8
            out = np.array(prog.run({"x": values})[0], copy=True)
            np.testing.assert_array_equal(out, values**2)
            r["checks"].append("MIGraphX-compiled-GPU-multiply-CPU-reference")
        x = torch.arange(17, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(17, dtype=torch.float32), atol=0, rtol=0
        )
        torch.cuda.synchronize()
        r["checks"].append("native-Torch-backward-after-optional-GPU-consumer")
        r["status"] = "passed"
    except BaseException as e:
        r.update(status="failed", error=repr(e))
        raise
    finally:
        a.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
