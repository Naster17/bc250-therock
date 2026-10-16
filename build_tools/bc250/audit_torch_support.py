#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only installed-stack audit and isolated, bounded Torch GPU probes.

An audit completing successfully does not mean every probe passed. Results retain
unsupported operations, exceptions, stderr and timeouts without warning filters.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

CASES = [
    "gemm-f32",
    "gemm-f64",
    "gemm-f16",
    "gemm-bf16",
    "gemm-lt",
    "gemm-int8",
    "scaled-fp8",
    "sparse-lt",
    "conv1d",
    "conv1d-g1",
    "conv1d-g1-fwd",
    "conv1d-g1-dx",
    "conv1d-g1-dw",
    "conv1d-g1-dx-direct",
    "conv1d-g1-dw-direct",
    "conv2d",
    "conv3d",
    "conv2d-f16",
    "conv2d-bf16",
    "depthwise-bf16",
    "conv-transpose",
    "batchnorm",
    "layernorm",
    "groupnorm",
    "pool-resize",
    "embedding",
    "loss-ce",
    "loss-ctc",
    "lstm",
    "fft-f32",
    "fft-f64",
    "fft-f16",
    "fft-bf16",
    "linalg-svd",
    "linalg-lstsq",
    "linalg-solve",
    "sparse-csr",
    "sparse-csc",
    "sparse-bsr",
    "sparse-solve",
    "rng",
    "sdpa-math",
    "sdpa-efficient",
    "sdpa-flash",
    "sdpa-tiled-head256",
    "adamw-fused",
    "amp",
    "compile",
    "graph",
    "distributed",
    "profiler",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(output: Path) -> dict:
    import torch

    root = Path("/workspace/build/dist/rocm")
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1013"
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "torch_path": torch.__file__,
        "architecture": props.gcnArchName,
        "hip_multiprocessors": props.multi_processor_count,
        "total_memory": props.total_memory,
        "config": torch.__config__.show(),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("MIOPEN_", "HSA_ENABLE_", "PYTORCH_ROCM_ARCH"))
        },
        "library_loads": {},
        "build_flags": {},
        "databases": {},
    }
    report["optional_python_packages"] = {}
    for distribution, module in [
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("torchao", "torchao"),
        ("bitsandbytes", "bitsandbytes"),
        ("flash-attn", "flash_attn"),
        ("flash-linear-attention", "fla"),
        ("causal-conv1d", "causal_conv1d"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("triton", "triton"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
    ]:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        spec = importlib.util.find_spec(module)
        report["optional_python_packages"][distribution] = {
            "version": version,
            "origin": str(spec.origin) if spec else None,
        }
    flags = [
        "_has_magma",
        "_has_cusparselt",
        "_has_hipsolver",
        "_has_lapack",
        "_has_mkldnn",
        "_has_cuda",
    ]
    report["torch_flags"] = {flag: getattr(torch._C, flag, None) for flag in flags}
    report["backend_flags"] = {
        "distributed": torch.distributed.is_available(),
        "miopen_version": torch.backends.cudnn.version(),
        "builtin_flash": torch.backends.cuda.is_flash_attention_available(),
        "tf32": torch.cuda.is_tf32_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "preferred_blas": str(torch.backends.cuda.preferred_blas_library()),
    }
    libraries = [
        "amdhip64",
        "hsa-runtime64",
        "amd_comgr",
        "hiprtc",
        "rocblas",
        "hipblas",
        "hipblaslt",
        "MIOpen",
        "hipfft",
        "rocfft",
        "hipsparse",
        "rocsparse",
        "hipsolver",
        "rocsolver",
        "hiprand",
        "rocrand",
        "hipsparselt",
        "hiptensor",
        "hipdnn_backend",
        "rccl",
        "MIOpenCKGroupedConv_gfx1013",
    ]
    handles = []
    for name in libraries:
        path = root / "lib" / ("lib" + name + ".so")
        row = {"path": str(path), "exists": path.exists()}
        if path.exists():
            row.update(resolved=str(path.resolve()), size_bytes=path.stat().st_size)
            try:
                handles.append(ctypes.CDLL(str(path)))
                row["dlopen"] = "ok"
            except OSError as error:
                row["dlopen_error"] = str(error)
        report["library_loads"][name] = row
    reader = shutil.which("readelf") or shutil.which("llvm-readelf")
    report["elf_dependencies"] = {}
    libtorch = Path(torch.__file__).parent / "lib/libtorch_hip.so"
    report["libtorch_hip_sha256"] = sha256(libtorch)
    if reader:
        for path in [
            libtorch,
            root / "lib/libMIOpen.so",
            root / "lib/libhipsparselt.so",
            root / "lib/libhiptensor.so",
            root / "lib/libhipdnn_backend.so",
        ]:
            result = subprocess.run(
                [reader, "-d", str(path)], capture_output=True, text=True, timeout=15
            )
            report["elf_dependencies"][str(path)] = {
                "returncode": result.returncode,
                "needed": re.findall(r"\(NEEDED\).*?\[(.*?)\]", result.stdout),
                "paths": [
                    s.strip()
                    for s in result.stdout.splitlines()
                    if "RUNPATH" in s or "RPATH" in s
                ],
            }
    db = root / "share/miopen/db"
    report["databases"] = {
        "directory": str(db),
        "exists": db.exists(),
        "files": (
            [str(p.relative_to(db)) for p in db.rglob("*") if p.is_file()]
            if db.exists()
            else []
        ),
        "gfx1013_assets": [
            str(p.relative_to(root)) for p in (root / "share/miopen").rglob("*gfx1013*")
        ],
    }
    # Only build flags, not arbitrary cached paths/credentials.
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    caches = [
        Path("/workspace/pytorch-work/build/CMakeCache.txt"),
        ws / "build/CMakeCache.txt",
    ]
    caches += list((ws / "build").glob("*/build/CMakeCache.txt"))
    caches += list((ws / "build").glob("*/*/build/CMakeCache.txt"))
    caches += list((ws / "build").glob("*/*/*/build/CMakeCache.txt"))
    keys = re.compile(
        r"^(USE_[A-Z0-9_]+|MIOPEN_[A-Z0-9_]+|PYTORCH_ROCM_ARCH|GPU_TARGETS|CMAKE_HIP_ARCHITECTURES|THEROCK_ENABLE_[A-Z0-9_]+|THEROCK_GFX1013_SCALAR_BACKENDS|THEROCK_AMDGPU_FAMILIES)$"
    )
    for cache in caches:
        if not cache.exists():
            continue
        selected = {}
        for line in cache.read_text().splitlines():
            match = re.match(r"([^:#]+):[^=]+=(.*)", line)
            if (
                match
                and keys.match(match[1])
                and not any(
                    word in match[1]
                    for word in ("PATH", "DIR", "KEY", "TOKEN", "PASSWORD")
                )
            ):
                selected[match[1]] = match[2]
        if selected:
            report["build_flags"][str(cache)] = selected
    registrations = []
    dispatch_keys = [
        "CUDA",
        "SparseCUDA",
        "SparseCsrCUDA",
        "QuantizedCUDA",
        "CompositeImplicitAutograd",
        "CompositeExplicitAutograd",
    ]
    for name in sorted(torch._C._dispatch_get_all_op_names()):
        registrations.append(
            {
                "operator": name,
                "keys": [
                    key
                    for key in dispatch_keys
                    if torch._C._dispatch_has_kernel_for_dispatch_key(name, key)
                ],
            }
        )
    (output.parent / "dispatch-registration.json").write_text(
        json.dumps(registrations, indent=2) + "\n"
    )
    report["registration_counts"] = {
        key: sum(key in row["keys"] for row in registrations) for key in dispatch_keys
    }
    report["registered_names"] = len(registrations)
    report["registration_warning"] = (
        "Registration is not gfx1013 coverage; composite/shape/dtype/device checks happen later."
    )
    report["loaded_maps"] = [
        s.strip()
        for s in Path("/proc/self/maps").read_text().splitlines()
        if ".so" in s and ("/workspace/" in s or "/opt/torch-" in s)
    ]
    return report


def probe(name: str) -> dict:
    import copy
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend

    torch.set_num_threads(2)
    torch.manual_seed(26016)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    details = {}

    def check(op, inputs, backward=False, tol=4e-4):
        cpu = [x.detach().clone().double().requires_grad_(backward) for x in inputs]
        # Preserve complex inputs in their independent CPU reference.
        cpu = [
            (
                x.detach().clone().to(torch.complex128).requires_grad_(backward)
                if x.is_complex()
                else c
            )
            for x, c in zip(inputs, cpu)
        ]
        gpu = [x.detach().cuda().requires_grad_(backward) for x in inputs]
        expected = op(*cpu)
        actual = op(*gpu)
        torch.testing.assert_close(
            actual.cpu().to(expected.dtype), expected, atol=tol, rtol=tol
        )
        details["max_absolute_error"] = float(
            (actual.detach().cpu().to(expected.dtype) - expected.detach()).abs().max()
        )
        if backward:
            grad = (
                torch.linspace(-0.4, 0.6, expected.numel())
                .reshape(expected.shape)
                .to(expected.dtype)
            )
            expected.backward(grad)
            actual.backward(grad.to(actual.device, actual.dtype))
            for c, g in zip(cpu, gpu):
                if c.grad is not None:
                    torch.testing.assert_close(
                        g.grad.cpu().to(c.grad.dtype), c.grad, atol=tol, rtol=tol
                    )
            details["backward_reference"] = True

    if name.startswith("gemm-") and name != "gemm-int8":
        dtype = {
            "f32": torch.float32,
            "f64": torch.float64,
            "f16": torch.float16,
            "bf16": torch.bfloat16,
            "lt": torch.float32,
        }[name[5:]]
        if name == "gemm-lt":
            torch.backends.cuda.preferred_blas_library("hipblaslt")
        check(
            lambda a, b: a @ b,
            [torch.randn(13, 17).to(dtype) * 0.2, torch.randn(17, 9).to(dtype) * 0.2],
            True,
            {torch.float64: 1e-10, torch.bfloat16: 0.025, torch.float16: 0.003}.get(
                dtype, 4e-4
            ),
        )
    elif name == "gemm-int8":
        a = torch.randint(-8, 9, (32, 32), dtype=torch.int8)
        b = torch.randint(-8, 9, (32, 32), dtype=torch.int8)
        out = torch._int_mm(a.cuda(), b.cuda()).cpu()
        torch.testing.assert_close(out, a.int() @ b.int(), atol=0, rtol=0)
    elif name == "scaled-fp8":
        dtype = torch.float8_e4m3fnuz
        a = (torch.randn(16, 32) * 0.2).cuda().to(dtype)
        b = (torch.randn(16, 32) * 0.2).cuda().to(dtype).T
        out = torch._scaled_mm(
            a,
            b,
            scale_a=torch.ones((), device="cuda"),
            scale_b=torch.ones((), device="cuda"),
            out_dtype=torch.float32,
        )
        torch.testing.assert_close(
            out.cpu(), a.float().cpu() @ b.float().cpu(), rtol=0.02, atol=0.02
        )
    elif name == "sparse-lt":
        a = torch.zeros(32, 64, dtype=torch.float16, device="cuda")
        a[:, ::4] = 1
        a[:, 1::4] = 2
        packed = torch._cslt_compress(a)
        out = torch._cslt_sparse_mm(
            packed, torch.ones(64, 32, dtype=torch.float16, device="cuda")
        )
        torch.testing.assert_close(
            out.cpu(), torch.full((32, 32), 48, dtype=torch.float16)
        )
    elif name.startswith("conv1d-g1-"):
        x = torch.randn(2, 4, 19) * 0.2
        w = torch.randn(6, 4, 3) * 0.2
        dx = "-dx" in name
        dw = "-dw" in name
        cpu = [x.double().requires_grad_(dx), w.double().requires_grad_(dw)]
        gpu = [x.cuda().requires_grad_(dx), w.cuda().requires_grad_(dw)]
        op = lambda a, b: F.conv1d(a, b, stride=2, padding=2, dilation=2, groups=1)
        expected = op(*cpu)
        actual = op(*gpu)
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=4e-4, rtol=4e-4
        )
        details["forward_reference"] = True
        if dx or dw:
            g = torch.linspace(-0.4, 0.6, expected.numel()).reshape(expected.shape)
            expected.backward(g.double())
            actual.backward(g.cuda())
            index = 0 if dx else 1
            torch.testing.assert_close(
                gpu[index].grad.cpu().double(), cpu[index].grad, atol=4e-4, rtol=4e-4
            )
            details["gradient_reference"] = "dx" if dx else "dw"
        details["miopen_gemm_env"] = os.environ.get("MIOPEN_DEBUG_CONV_GEMM", "unset")
    elif name.startswith("conv") or name == "depthwise-bf16":
        if name == "conv-transpose":
            check(
                lambda a, w: F.conv_transpose2d(a, w, stride=2, padding=1),
                [torch.randn(1, 4, 5, 7) * 0.2, torch.randn(4, 3, 3, 3) * 0.2],
                True,
            )
        elif name == "depthwise-bf16":
            check(
                lambda a, w: F.silu(F.conv1d(a, w, padding=3, groups=64)[..., :65]),
                [
                    (torch.randn(1, 64, 65) * 0.2).bfloat16(),
                    (torch.randn(64, 1, 4) * 0.2).bfloat16(),
                ],
                True,
                0.025,
            )
        else:
            rank = int(name[4])
            spatial = {1: (19,), 2: (9, 11), 3: (5, 7, 9)}[rank]
            dtype = (
                torch.float16
                if name.endswith("f16")
                else torch.bfloat16 if name.endswith("bf16") else torch.float32
            )
            op = getattr(F, f"conv{rank}d")
            groups = 1 if name.endswith("g1") else 2
            check(
                lambda a, w: op(a, w, stride=2, padding=2, dilation=2, groups=groups),
                [
                    (torch.randn(2, 4, *spatial) * 0.2).to(dtype),
                    (torch.randn(6, 4 // groups, *([3] * rank)) * 0.2).to(dtype),
                ],
                True,
                (
                    0.025
                    if dtype == torch.bfloat16
                    else 0.004 if dtype == torch.float16 else 4e-4
                ),
            )
    elif name == "batchnorm":
        check(
            lambda a: F.batch_norm(a, None, None, training=True),
            [torch.randn(2, 4, 7, 9)],
            True,
        )
    elif name == "layernorm":
        check(lambda a: F.layer_norm(a, (17,)), [torch.randn(2, 3, 17)], True)
    elif name == "groupnorm":
        check(lambda a: F.group_norm(a, 2), [torch.randn(2, 4, 17)], True)
    elif name == "pool-resize":
        check(
            lambda a: F.interpolate(
                F.avg_pool2d(a, 3, 2, 1, count_include_pad=False),
                size=(7, 11),
                mode="bilinear",
                align_corners=False,
            ),
            [torch.randn(2, 4, 9, 13).contiguous(memory_format=torch.channels_last)],
            True,
        )
    elif name == "embedding":
        ids = torch.tensor([0, 2, 2, 7, 2, 1])
        check(lambda w: F.embedding(ids.to(w.device), w), [torch.randn(11, 7)], True)
    elif name == "loss-ce":
        labels = torch.tensor([0, 2, -100, 7])
        check(
            lambda a: F.cross_entropy(a, labels.to(a.device)),
            [torch.randn(4, 11)],
            True,
        )
    elif name == "loss-ctc":
        targets = torch.tensor([1, 2, 3, 1], dtype=torch.long)
        lengths = torch.tensor([8, 8])
        target_lengths = torch.tensor([2, 2])
        check(
            lambda a: F.ctc_loss(
                a.log_softmax(-1), targets.to(a.device), lengths, target_lengths
            ),
            [torch.randn(8, 2, 5)],
            True,
        )
    elif name == "lstm":
        cpu = torch.nn.LSTM(3, 4, batch_first=True).double()
        gpu = copy.deepcopy(cpu).float().cuda()
        x = torch.randn(2, 5, 3)
        xc = x.double().requires_grad_()
        xg = x.cuda().requires_grad_()
        yc = cpu(xc)[0]
        yg = gpu(xg)[0]
        torch.testing.assert_close(yg.cpu().double(), yc, atol=4e-4, rtol=4e-4)
        yc.sum().backward()
        yg.sum().backward()
        torch.testing.assert_close(
            xg.grad.cpu().double(), xc.grad, atol=4e-4, rtol=4e-4
        )
        for a, b in zip(gpu.parameters(), cpu.parameters()):
            torch.testing.assert_close(
                a.grad.cpu().double(), b.grad, atol=4e-4, rtol=4e-4
            )
    elif name.startswith("fft-"):
        dtype = {
            "f32": torch.float32,
            "f64": torch.float64,
            "f16": torch.float16,
            "bf16": torch.bfloat16,
        }[name[4:]]
        check(
            lambda a: torch.fft.rfft2(a).abs().square().mean(),
            [torch.randn(2, 8, 16).to(dtype) * 0.1],
            True,
            1e-10 if dtype == torch.float64 else 0.005,
        )
    elif name == "linalg-svd":
        check(lambda a: torch.linalg.svdvals(a), [torch.randn(9, 5)], True, 0.001)
    elif name == "linalg-lstsq":
        a = torch.randn(13, 5)
        b = torch.randn(13, 2)
        expected = torch.linalg.lstsq(a.double(), b.double(), driver="gels").solution
        actual = torch.linalg.lstsq(a.cuda(), b.cuda(), driver="gels").solution
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=5e-4, rtol=5e-4
        )
    elif name == "linalg-solve":
        a = torch.randn(7, 7)
        a = a @ a.T + torch.eye(7) * 2
        check(torch.linalg.solve, [a, torch.randn(7, 3)], True)
    elif name.startswith("sparse-"):
        a = torch.eye(8) * 3 + torch.diag(torch.ones(7), 1)
        b = torch.randn(8, 3)
        if name == "sparse-solve":
            out = torch.sparse.spsolve(a.cuda().to_sparse_csr(), b[:, 0].cuda())
            torch.testing.assert_close(
                out.cpu(), torch.linalg.solve(a, b[:, 0]), atol=4e-4, rtol=4e-4
            )
        else:
            ag = a.cuda()
            sparse = {
                "sparse-csr": lambda: ag.to_sparse_csr(),
                "sparse-csc": lambda: ag.to_sparse_csc(),
                "sparse-bsr": lambda: ag.to_sparse_bsr((2, 2)),
            }[name]()
            out = torch.sparse.mm(sparse, b.cuda())
            torch.testing.assert_close(out.cpu(), a @ b, atol=4e-4, rtol=4e-4)
    elif name == "rng":
        torch.cuda.manual_seed_all(891)
        state = torch.cuda.get_rng_state()
        a = torch.randn(103, device="cuda")
        torch.cuda.set_rng_state(state)
        b = torch.randn(103, device="cuda")
        assert torch.equal(a, b) and bool(torch.isfinite(a).all())
        details["exact_rng_replay"] = True
    elif name.startswith("sdpa-"):
        q = torch.randn(1, 2, 17, 256 if name == "sdpa-tiled-head256" else 32) * 0.2
        cpu = [
            t.double().requires_grad_()
            for t in (q, torch.randn_like(q), torch.randn_like(q))
        ]
        gpu = [t.detach().float().cuda().requires_grad_() for t in cpu]
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(*cpu, is_causal=True)
        if name == "sdpa-tiled-head256":
            from torch.nn.attention._gfx1013 import scaled_dot_product_attention

            actual = scaled_dot_product_attention(*gpu, is_causal=True)
        else:
            backend = {
                "sdpa-math": SDPBackend.MATH,
                "sdpa-efficient": SDPBackend.EFFICIENT_ATTENTION,
                "sdpa-flash": SDPBackend.FLASH_ATTENTION,
            }[name]
            with sdpa_kernel(backend):
                actual = F.scaled_dot_product_attention(*gpu, is_causal=True)
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=4e-4, rtol=4e-4
        )
        expected.sum().backward()
        actual.sum().backward()
        for a, b in zip(gpu, cpu):
            torch.testing.assert_close(
                a.grad.cpu().double(), b.grad, atol=4e-4, rtol=4e-4
            )
    elif name == "adamw-fused":
        cpu = torch.randn(31, 17, requires_grad=True)
        gpu = cpu.detach().clone().cuda().requires_grad_()
        cpu.grad = torch.randn_like(cpu)
        gpu.grad = cpu.grad.cuda()
        torch.optim.AdamW([cpu], lr=0.001).step()
        torch.optim.AdamW([gpu], lr=0.001, fused=True).step()
        torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-6, rtol=1e-6)
    elif name == "amp":
        w = torch.randn(17, 9, device="cuda", requires_grad=True)
        a = torch.randn(13, 17, device="cuda")
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        optimizer = torch.optim.SGD([w], lr=0.01)
        with torch.autocast("cuda", dtype=torch.float16):
            product = a @ w
            loss = product.square().mean()
        assert product.dtype == torch.float16
        details["product_dtype"] = str(product.dtype)
        details["loss_dtype"] = str(loss.dtype)
        before = w.detach().clone()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        assert bool(torch.isfinite(w).all()) and not torch.equal(w, before)
        details["contract_only"] = (
            "finite AMP update; existing CPU-gradient AMP qualification retained"
        )
    elif name == "compile":
        op = lambda a: F.layer_norm(F.gelu(a), (17,)).square()
        compiled = torch.compile(op, fullgraph=True)
        a = torch.randn(2, 3, 17, device="cuda", requires_grad=True)
        b = a.detach().clone().requires_grad_()
        y = compiled(a)
        expected = op(b)
        torch.testing.assert_close(y, expected, atol=4e-4, rtol=4e-4)
        y.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(a.grad, b.grad, atol=5e-4, rtol=5e-4)
    elif name == "graph":
        a = torch.randn(31, device="cuda")
        b = torch.empty_like(a)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.add(a, 2, out=b)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.add(a, 2, out=b)
        a.fill_(3)
        graph.replay()
        torch.testing.assert_close(b.cpu(), torch.full((31,), 5.0))
    elif name == "distributed":
        if not torch.distributed.is_available():
            raise NotImplementedError(
                "Torch USE_DISTRIBUTED=OFF; RCCL/Gloo not built into this runtime"
            )
        details["available"] = True
        raise NotImplementedError(
            "Distributed availability requires separate process-group qualification"
        )
    elif name == "profiler":
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            a = torch.randn(17, device="cuda")
            b = a.sin()
            torch.cuda.synchronize()
        events = prof.events()
        details["events"] = len(events)
        details["gpu_events"] = sum("CUDA" in str(e.device_type) for e in events)
        if not details["gpu_events"]:
            raise NotImplementedError(
                "No GPU events: Torch USE_KINETO=OFF; independent rocprofiler is separate"
            )
    else:
        raise ValueError(name)
    torch.cuda.synchronize()
    return details


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--only", nargs="+", choices=CASES)
    args = ap.parse_args()
    if args.case:
        result = {"case": args.case}
        try:
            result.update(status="passed", details=probe(args.case))
        except Exception as error:
            result.update(
                status=(
                    "unsupported" if isinstance(error, NotImplementedError) else "error"
                ),
                exception=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    result = {
        "inventory": inventory(args.output),
        "source_sha256": sha256(Path(__file__)),
        "probes": [],
        "status": "running",
        "meaning": "Completion is an audit result, not universal support or an all-probes-pass assertion.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in args.only or CASES:
        out = args.output.parent / (case + ".json")
        log = args.output.parent / (case + ".log")
        start = time.monotonic()
        with log.open("w") as stream:
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "--output", str(out), "--case", case],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=45 if case == "compile" else 25,
                    env=(
                        dict(os.environ, MIOPEN_DEBUG_CONV_GEMM="0")
                        if case.endswith("-direct")
                        else None
                    ),
                )
                row = (
                    json.loads(out.read_text())
                    if out.exists()
                    else {"case": case, "status": "process_failure"}
                )
                row.update(returncode=proc.returncode, seconds=time.monotonic() - start)
            except subprocess.TimeoutExpired:
                row = {
                    "case": case,
                    "status": "timeout",
                    "seconds": time.monotonic() - start,
                }
        result["probes"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(case, row["status"], row.get("error", "")[:180], flush=True)
        if row.get("returncode", 0) < 0 or row["status"] == "timeout":
            raise RuntimeError(
                "Stopped after child crash/timeout; retained audit evidence"
            )
    result["status"] = (
        "completed_with_findings"
        if any(p["status"] != "passed" for p in result["probes"])
        else "completed"
    )
    result["counts"] = {
        status: sum(p["status"] == status for p in result["probes"])
        for status in sorted({p["status"] for p in result["probes"]})
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
