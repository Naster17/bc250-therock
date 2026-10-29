#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate ordinary installed opt-in activation or retained plain runtime defaults."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b077")


def public_same_process():
    import torch
    import torch.nn.functional as F

    lib = C.CDLL("libMIOpen.so.1")
    P = C.c_void_p
    I = C.c_int
    Z = C.c_size_t

    def api(name, types, *args):
        fn = getattr(lib, "miopen" + name)
        fn.argtypes = types
        fn.restype = I
        status = fn(*args)
        assert status == 0, (name, status)

    handle = P()
    conv = P()
    descs = []
    x = (
        torch.arange(2 * 16 * 32 * 34, dtype=torch.float32).reshape(2, 16, 32, 34) % 19
        - 9
    ) / 11
    w = (
        torch.arange(16 * 16 * 3 * 3, dtype=torch.float32).reshape(16, 16, 3, 3) % 17
        - 8
    ) / 11
    ref = F.conv2d(x.double(), w.double(), padding=1)
    a = x.cuda()
    b = w.cuda()
    out = torch.full_like(ref, float("nan"), dtype=torch.float32, device="cuda")
    try:
        api("Create", [C.POINTER(P)], C.byref(handle))
        api("SetStream", [P, P], handle, P(torch.cuda.current_stream().cuda_stream))
        for tensor in (a, b, out):
            d = P()
            api("CreateTensorDescriptor", [C.POINTER(P)], C.byref(d))
            descs.append(d)
            api("Set4dTensorDescriptor", [P, I, I, I, I, I], d, 1, *tensor.shape)
        xd, wd, yd = descs
        api("CreateConvolutionDescriptor", [C.POINTER(P)], C.byref(conv))
        api("InitConvolutionDescriptor", [P] + [I] * 7, conv, 0, 1, 1, 1, 1, 1, 1)
        space = Z()
        api(
            "ConvolutionForwardGetSolutionWorkspaceSize",
            [P] * 5 + [C.c_uint64, C.POINTER(Z)],
            handle,
            wd,
            xd,
            conv,
            yd,
            137,
            C.byref(space),
        )
        assert space.value < 64 * 1024 * 1024
        work = torch.empty(max(space.value, 1), dtype=torch.uint8, device="cuda")
        api(
            "ConvolutionForwardCompileSolution",
            [P] * 5 + [C.c_uint64],
            handle,
            wd,
            xd,
            conv,
            yd,
            137,
        )
        api(
            "ConvolutionForwardImmediate",
            [P] * 9 + [Z, C.c_uint64],
            handle,
            wd,
            P(b.data_ptr()),
            xd,
            P(a.data_ptr()),
            conv,
            yd,
            P(out.data_ptr()),
            P(work.data_ptr()) if space.value else P(),
            space.value,
            137,
        )
        torch.cuda.synchronize()
        error = (out.cpu().double() - ref).abs()
        assert torch.isfinite(out).all() and bool(
            (error <= 2e-5 + 5e-5 * ref.abs()).all()
        )
        torch.testing.assert_close(a.cpu(), x, rtol=0, atol=0)
        torch.testing.assert_close(b.cpu(), w, rtol=0, atol=0)
        return float(error.max())
    finally:
        if conv.value:
            api("DestroyConvolutionDescriptor", [P], conv)
        for d in descs:
            api("DestroyTensorDescriptor", [P], d)
        if handle.value:
            api("Destroy", [P], handle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plain", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    r = dict(status="running", checks=[], probes={})
    try:
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(1)
        torch.manual_seed(7801)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert torch.__file__ == "/workspace/pytorch-work/torch/__init__.py"
        if args.plain:
            assert "MIOPEN_CK_LIB_PATH" not in os.environ
            r["checks"].append(
                "plain-runtime-native-Torch-origin-without-plugin-environment"
            )
        else:
            assert Path(os.environ["MIOPEN_CK_LIB_PATH"]) == ROOT / "lib"
            assert os.environ.get("BC250_CK_MARKER") == "installed"
            # The installed artifact has its own immutable read-only mount.
            mounts = Path("/proc/self/mountinfo").read_text().splitlines()
            assert any(
                line.split()[4] == str(ROOT) and "ro" in line.split()[5].split(",")
                for line in mounts
            )
            r["checks"].append(
                "ordinary-wrapper-installed-path-env-readonly-and-native-origin"
            )
            r["same_process_max_abs_error"] = public_same_process()
            r["checks"].append("public137-in-entry-verified-process-matches-CPU-double")
            db = args.output.parent / "example-db"
            shutil.copytree(
                ROOT
                / "examples"
                / ("rccl" if torch.distributed.is_nccl_available() else "daily"),
                db,
            )
            env = dict(
                os.environ,
                MIOPEN_USER_DB_PATH=str(db),
                MIOPEN_CUSTOM_CACHE_DIR=str(args.output.parent / "private-cache"),
                MIOPEN_LOG_LEVEL="6",
                MIOPEN_FIND_ENFORCE="NONE",
            )
            for kind, extra in [
                ("b072", []),
                ("b073", [str(ROOT / "lib/libMIOpenCKGroupedConv_gfx1013.so")]),
                ("b076", ["--diagnostic"]),
            ]:
                proc = subprocess.run(
                    [str(ROOT / "probes" / ("probe-" + kind)), *extra],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                (args.output.parent / (kind + "-stdout.log")).write_text(proc.stdout)
                (args.output.parent / (kind + "-stderr.log")).write_text(proc.stderr)
                proc.check_returncode()
                value = json.loads(proc.stdout.strip().splitlines()[-1])
                assert value["status"] == "passed"
                if kind == "b072":
                    assert value["executions"] == 32 and value["refusals"] == 152
                if kind == "b073":
                    assert (
                        value["executions"] == 256
                        and value["api_refusals"] == 122
                        and value["runtime_refusals"] == 246
                    )
                if kind == "b076":
                    assert (
                        len(value["rows"]) == 2
                        and "Search Started:" not in proc.stderr
                        and "Perf Db: record loaded:" in proc.stderr
                    )
                r["probes"][kind] = value
                r["checks"].append("installed-" + kind + "-CPU-reference-and-contracts")
        for groups in (1, 3):
            for f in (1, 3):
                values = [
                    torch.randn(2, groups * 4, 5, 7) / 8,
                    torch.randn(groups * 4, 4, f, f) / 8,
                    torch.randn(groups * 4) / 8,
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
                    f"ordinary-Torch-conv-output-all-first-gradients-g{groups}-f{f}"
                )
        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        r["checks"].append("native-backward-after-installed-or-plain-operations")
        assert len(r["checks"]) == (6 if args.plain else 10)
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
