#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate ordinary installed B07.12 FP32/FP16 activation or plain defaults.

Installed runs (bc250-ck with BC250_CK_MARKER=installed) perform 15 checks:
wrapper environment/read-only mount/native Torch origin; entry --check JSON
with 27 mapped runtime identities and libMIOpen.so.1 from the installed root;
same-process public solver137 FP32 matching CPU double; the eight-key mixed
example seed (2 FP32 + 6 FP16 per cohort); six probe batteries (retained FP32
b072/b073/b076 plus FP16 b0710_public/b0710_half/b0711) against the B07.12
reference counts; four ordinary Torch conv output/all-first-gradient fixtures;
and the native backward fixture. Plain runs (bc250-torch --plain) perform 6
checks: native Torch origin without plugin environment, the same four gradient
fixtures and native backward.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b0712")


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
            # Fresh format2 worker: the entry verifies every payload hash and
            # the actual mapped runtime identities before printing --check.
            # This validator already runs inside the entry's run worker, whose
            # LD_LIBRARY_PATH prepends the private runtime/sysdeps paths. A
            # nested check inheriting that polluted path maps extra ROCm
            # dependencies (rocprofiler-sdk/dw/lzma) and drifts from the pinned
            # 27-library identity. Strip the worker prefix back to the container
            # base so the nested check reproduces the fresh-container identity.
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "ck_entry", ROOT / "bin/bc250-ck.py"
            )
            entry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entry)
            worker_prefix = (
                str(ROOT / "runtime")
                + ":"
                + str(entry.PREFIX / "lib")
                + ":"
                + str(entry.PREFIX / "lib/rocm_sysdeps/lib")
                + ":"
            )
            base_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if base_ld.startswith(worker_prefix):
                base_ld = base_ld[len(worker_prefix) :]
            proc = subprocess.run(
                [sys.executable, str(ROOT / "bin/bc250-ck.py"), "check"],
                env=dict(os.environ, LD_LIBRARY_PATH=base_ld),
                capture_output=True,
                text=True,
                timeout=240,
            )
            (args.output.parent / "entry-check-stdout.log").write_text(proc.stdout)
            (args.output.parent / "entry-check-stderr.log").write_text(proc.stderr)
            proc.check_returncode()
            identity = json.loads(proc.stdout.strip().splitlines()[-1])
            assert identity["status"] == "passed"
            assert identity["artifact"] == ROOT.name
            assert identity["architecture"] == "gfx1013:xnack-"
            assert identity["api"] == 5
            assert len(identity["runtime_libraries"]) == 27
            assert identity["runtime_libraries"]["libMIOpen.so.1"]["path"] == str(
                ROOT / "runtime/libMIOpen.so.1"
            )
            r["entry_check"] = identity
            r["checks"].append(
                "entry-check-27-mapped-runtime-identities-and-private-MIOpen-origin"
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
            seeded = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in db.glob("*.txt")
            }
            assert len(seeded) == 2
            for file in db.glob("*.txt"):
                lines = file.read_text().splitlines()
                assert (
                    len(lines) == 8
                    and sum("FP16" in x for x in lines) == 6
                    and sum("FP32" in x for x in lines) == 2
                )
            r["checks"].append("installed-eight-key-mixed-FP32-FP16-example-seed")
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
                ("b0710_public", []),
                ("b0710_half", [str(ROOT / "lib/libMIOpenCKGroupedConv_gfx1013.so")]),
                ("b0711", ["--diagnostic"]),
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
                    assert (
                        value["executions"] == 32
                        and value["refusals"] == 152
                        and value["guards"]
                    )
                if kind == "b073":
                    assert (
                        value["fixtures"] == 80
                        and value["executions"] == 256
                        and value["api_refusals"] == 122
                        and value["runtime_refusals"] == 246
                        and len(value["kernel_counts"]) == 6
                    )
                if kind in ("b076", "b0711"):
                    count = 2 if kind == "b076" else 6
                    assert (
                        len(value["rows"]) == count
                        and value["guards"]
                        and "Search Started:" not in proc.stderr
                        and proc.stderr.count(
                            "Perf Db: record loaded: ConvHipImplicitGemmGroupFwdXdlops"
                        )
                        >= count
                    )
                if kind == "b0710_public":
                    assert (
                        value["executions"] == 32
                        and value["refusals"] == 0
                        and value["guards"]
                    )
                if kind == "b0710_half":
                    assert (
                        value["fixtures"] == 88
                        and value["executions"] >= 200
                        and value["api_refusals"] == 184
                        and value["runtime_refusals"] == 477
                        and len(value["kernel_counts"]) == 9
                    )
                    assert all(n > 0 for n in value["kernel_counts"].values())
                    assert value["guards"]
                r["probes"][kind] = value
                r["checks"].append("installed-" + kind + "-CPU-reference-and-contracts")
            assert {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in db.glob("*.txt")
            } == seeded, "probes changed seeded tuning records"
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
        assert len(r["checks"]) == (6 if args.plain else 15)
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
