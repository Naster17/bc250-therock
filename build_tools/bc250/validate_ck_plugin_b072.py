#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Force the private MIOpen DL solver and verify explicit refusals and coexistence."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    root = ws / "tmp/ck-plugin-b072/build-r8"
    assert Path(os.environ["MIOPEN_CK_LIB_PATH"]) == root
    r = dict(status="running", checks=[], native={})
    try:
        lib = C.CDLL(str(root / "libMIOpenCKGroupedConv_gfx1013.so"))
        audit = json.loads((ws / "logs/ck-plugin-b071-source-audit.json").read_text())
        assert all(getattr(lib, s) for s in audit["required_symbols"])
        assert lib.ck_impl_get_api_version() == 5
        fill = lib.ck_impl_fwd_get_all_kernel_type_strings
        fill.argtypes = [C.POINTER(C.c_void_p)]
        fill.restype = C.c_int
        size = lib.ck_impl_kernel_list_size
        size.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        size.restype = C.c_int
        get = lib.ck_impl_kernel_list_get
        get.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_char_p)]
        get.restype = C.c_int
        free = lib.ck_impl_kernel_list_free
        free.argtypes = [C.c_void_p]
        free.restype = None
        handle = C.c_void_p()
        assert fill(C.byref(handle)) == 0
        count = C.c_size_t()
        assert size(handle, C.byref(count)) == 0 and count.value == 6
        ids = []
        for index in range(count.value):
            value = C.c_char_p()
            assert get(handle, index, C.byref(value)) == 0
            ids.append(value.value.decode())
        value = C.c_char_p(b"unchanged")
        assert (
            get(handle, count.value, C.byref(value)) == 2
            and value.value == b"unchanged"
        )
        assert size(None, C.byref(count)) == 1
        free(handle)
        assert len(set(ids)) == 6 and all("dl" in s.lower() for s in ids)
        r["kernel_ids"] = ids
        r["checks"].append("api5-required66-symbols-and-list-contract")
        for mode in ("private", "without-private"):
            env = os.environ.copy()
            cmd = [str(root / "probe")]
            if mode == "without-private":
                env.pop("MIOPEN_CK_LIB_PATH", None)
                cmd += ["--negative"]
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, check=False, timeout=180
            )
            (args.output.parent / (mode + "-stdout.log")).write_text(proc.stdout)
            (args.output.parent / (mode + "-stderr.log")).write_text(proc.stderr)
            proc.check_returncode()
            result = json.loads(proc.stdout.strip().splitlines()[-1])
            assert result["status"] == "passed"
            if mode == "private":
                assert (
                    result["executions"] == 32
                    and result["refusals"] == 152
                    and result["guards"]
                )
            else:
                assert result["executions"] == 0 and result["refusals"] == 32
            r["native"][mode] = result
            r["checks"].append(mode + "-forced-solver-and-refusals")
        import torch
        import torch.nn.functional as F

        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        torch.set_num_threads(1)
        torch.manual_seed(7201)
        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        r["checks"].append("native-torch-backward-after-private-plugin")
        for groups in (1, 3):
            for f in (1, 3):
                values = [
                    torch.randn(2, 4 * groups, 5, 7) / 8,
                    torch.randn(4 * groups, 4, f, f) / 8,
                    torch.randn(4 * groups) / 8,
                ]
                cpu = [v.double().requires_grad_() for v in values]
                gpu = [v.cuda().requires_grad_() for v in values]
                a = F.conv2d(*cpu, padding=f // 2, groups=groups)
                b = F.conv2d(*gpu, padding=f // 2, groups=groups)
                cot = torch.linspace(
                    -0.5, 0.5, a.numel(), dtype=torch.float64
                ).reshape_as(a)
                (a * cot).sum().backward()
                (b * cot.float().cuda()).sum().backward()
                torch.cuda.synchronize()
                torch.testing.assert_close(b.cpu().double(), a, rtol=3e-4, atol=2e-5)
                for ref, actual in zip(cpu, gpu):
                    torch.testing.assert_close(
                        actual.grad.cpu().double(), ref.grad, rtol=3e-4, atol=2e-5
                    )
                r["checks"].append(
                    f"torch-conv-output-all-first-gradients-g{groups}-f{f}"
                )
        r["status"] = "passed"
        assert len(r["checks"]) == 8
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
