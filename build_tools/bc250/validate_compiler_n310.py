#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Model-independent Inductor training contracts against CPU FP64 and AdamW."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch._inductor import metrics


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(17, 37)
        self.down = nn.Linear(37, 17)
        self.norm = nn.LayerNorm(17)

    def forward(self, x):
        return self.norm(x + self.down(F.silu(self.up(x))))


class Gated(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(17, 37)
        self.up = nn.Linear(17, 37)
        self.down = nn.Linear(37, 17)
        self.scale = nn.Parameter(torch.linspace(0.8, 1.2, 17))

    def forward(self, x):
        y = x + self.down(F.silu(self.gate(x)) * self.up(x))
        return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5) * self.scale


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=2, dilation=2, groups=2)
        self.norm = nn.GroupNorm(2, 8)
        self.head = nn.Linear(8 * 2 * 3, 7)

    def forward(self, x):
        y = F.gelu(self.norm(self.conv(x)))
        return self.head(F.adaptive_avg_pool2d(y, (2, 3)).flatten(1))


class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(4, momentum=0.2)
        self.head = nn.Linear(4, 7)

    def forward(self, x):
        return self.head(F.silu(self.norm(x)).mean((2, 3)))


MODELS = dict(dense=Dense, gated=Gated, cnn=CNN, batchnorm=BatchNorm)


def tensor_sha(x):
    return hashlib.sha256(
        x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def compare(actual, expected, dtype):
    actual = actual.detach().cpu().double()
    expected = expected.detach().double()
    assert bool(actual.isfinite().all()) and bool(expected.isfinite().all())
    rtol, atol, limit = {
        torch.float32: (8e-4, 2e-5, 5e-4),
        torch.float16: (0.02, 8e-4, 0.006),
        torch.bfloat16: (0.08, 0.006, 0.04),
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
    assert relative < limit, (relative, limit)
    return relative


def finite_differences(base, inputs, upstream, cpu):
    # Separate perturbed FP64 executions validate the CPU autograd oracle itself.
    probes = []
    for kind, index in (("input", 0), ("input", inputs.numel() // 2), ("parameter", 0)):
        values = []
        for sign in (-1, 1):
            module = copy.deepcopy(base).double()
            x = inputs.detach().double().clone()
            with torch.no_grad():
                target = x if kind == "input" else next(module.parameters())
                target.reshape(-1)[index] += sign * 1e-5
                values.append(float((module(x) * upstream).sum()))
        approximate = (values[1] - values[0]) / 2e-5
        gradient = cpu[0].grad if kind == "input" else next(cpu[1].parameters()).grad
        exact = float(gradient.reshape(-1)[index])
        assert abs(approximate - exact) < 2e-7 + abs(exact) * 2e-5, (
            kind,
            approximate,
            exact,
        )
        probes.append(
            dict(kind=kind, index=index, exact=exact, finite_difference=approximate)
        )
    return probes


def optimizer_check(module):
    actual = list(module.parameters())
    expected = [nn.Parameter(p.detach().cpu().clone()) for p in actual]
    for a, b in zip(actual, expected):
        b.grad = a.grad.detach().cpu().clone()
    options = dict(lr=1e-3, weight_decay=0.01, foreach=False)
    gpu = torch.optim.AdamW(actual, **options)
    cpu = torch.optim.AdamW(expected, **options)
    gpu.step()
    cpu.step()
    maximum = 0.0
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.detach().cpu(), b.detach(), atol=2e-7, rtol=1e-5)
        maximum = max(maximum, float((a.detach().cpu() - b.detach()).abs().max()))
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                gpu.state[a][key].cpu(), cpu.state[b][key], atol=1e-10, rtol=2e-5
            )
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(MODELS), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert (
        torch.version.hip
        and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    )
    assert not torch._dynamo.config.suppress_errors
    report = dict(
        status="running",
        family=args.family,
        checks=[],
        results=[],
        compilation=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference="CPU FP64 forward/input/all-parameter gradients with quantized upstream; finite differences; independent CPU FP32 AdamW of each GPU gradient",
        scope="Synthetic modules, fullgraph Inductor forward/backward; no model packages, compiled optimizer, graph capture, stochastic layer, full application or speed claim",
    )
    save = lambda: args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for precision_id, dtype in enumerate(
            (torch.float32, torch.float16, torch.bfloat16)
        ):
            torch.manual_seed(31010 + precision_id)
            base = MODELS[args.family]().train()
            module = copy.deepcopy(base).cuda()
            graphs = []
            before = metrics.generated_kernel_count

            def backend(graph, inputs):
                graphs.append(graph.code)
                return torch._inductor.compile(graph, inputs)

            compiled = torch.compile(
                module, backend=backend, fullgraph=True, dynamic=False
            )
            for shape_id in range(2):
                if args.family in ("dense", "gated"):
                    raw = torch.randn((2, 9, 17) if shape_id == 0 else (3, 7, 34)) * 0.2
                    inputs = raw if shape_id == 0 else raw[..., ::2]
                    gpu_raw = raw.cuda()
                    gpu_input = gpu_raw if shape_id == 0 else gpu_raw[..., ::2]
                else:
                    raw = (
                        torch.randn((2, 4, 7, 9) if shape_id == 0 else (3, 4, 9, 11))
                        * 0.2
                    )
                    inputs = (
                        raw.contiguous(memory_format=torch.channels_last)
                        if shape_id
                        else raw
                    )
                    gpu_input = inputs.cuda()
                exact = inputs.detach().double().clone().requires_grad_()
                oracle = copy.deepcopy(base).double()
                expected = oracle(exact)
                output_dtype = (
                    dtype if args.family in ("cnn", "batchnorm") else torch.float32
                )
                upstream = (
                    torch.linspace(-0.7, 0.9, expected.numel())
                    .reshape(expected.shape)
                    .div(expected.numel())
                    .to(output_dtype)
                )
                expected.backward(upstream.double())
                if precision_id == shape_id == 0:
                    report["finite_difference_probes"] = finite_differences(
                        base, inputs, upstream.double(), (exact, oracle)
                    )
                    report["checks"].append("cpu-fp64-oracle-finite-differences")
                for mode in ("eager", "compiled"):
                    module.load_state_dict(base.state_dict())
                    module.zero_grad(set_to_none=True)
                    x = gpu_input.detach().requires_grad_()
                    execute = module if mode == "eager" else compiled
                    with torch.autocast(
                        "cuda", dtype=dtype, enabled=dtype != torch.float32
                    ):
                        actual = execute(x)
                    assert actual.dtype == output_dtype, (actual.dtype, output_dtype)
                    actual.backward(upstream.cuda())
                    errors = {
                        "output": compare(actual, expected, dtype),
                        "input_gradient": compare(x.grad, exact.grad, dtype),
                    }
                    for (name, p), (cpu_name, reference) in zip(
                        module.named_parameters(), oracle.named_parameters()
                    ):
                        assert (
                            name == cpu_name
                            and p.grad is not None
                            and float(p.grad.norm()) > 0
                        )
                        errors[name] = compare(p.grad, reference.grad, dtype)
                    for name, buffer in module.named_buffers():
                        reference = dict(oracle.named_buffers())[name]
                        torch.testing.assert_close(
                            buffer.cpu().double(),
                            reference.double(),
                            atol=2e-6,
                            rtol=2e-5,
                        )
                    update_error = optimizer_check(module)
                    torch.cuda.synchronize()
                    report["results"].append(
                        dict(
                            mode=mode,
                            precision=str(dtype),
                            shape=list(inputs.shape),
                            input_stride=list(x.stride()),
                            input_sha256=tensor_sha(inputs),
                            max_relative_l2=max(errors.values()),
                            relative_l2=errors,
                            parameter_count=len(list(module.parameters())),
                            buffer_count=len(list(module.buffers())),
                            cpu_adamw_max_parameter_error=update_error,
                        )
                    )
                    report["checks"].append(
                        f"{dtype}-shape{shape_id}-{mode}-output-all-gradients-buffers-cpu-adamw"
                    )
                    save()
                if precision_id == shape_id == 0:
                    module.load_state_dict(base.state_dict())
                    module.zero_grad(set_to_none=True)
                    x = gpu_input.detach().requires_grad_()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        out = compiled(x)
                        out.backward(upstream.cuda())
                        torch.cuda.synchronize()
                    compare(out, expected, dtype)
                    compare(x.grad, exact.grad, dtype)
                    for p, c in zip(module.parameters(), oracle.parameters()):
                        compare(p.grad, c.grad, dtype)
                    path = args.output.parent / "compiled-gpu-trace.json"
                    prof.export_chrome_trace(str(path))
                    events = json.loads(path.read_text())["traceEvents"]
                    kernels = [
                        e
                        for e in events
                        if e.get("cat") == "kernel" and e.get("ph") == "X"
                    ]
                    assert kernels and any(
                        "triton" in e.get("name", "").lower() for e in kernels
                    )
                    report["gpu_kernel_names"] = sorted({e["name"] for e in kernels})
                    report["gpu_kernel_count"] = len(kernels)
                    report["checks"].append(
                        "profiled-compiled-gpu-kernels-reference-replay"
                    )
                    save()
            generated = metrics.generated_kernel_count - before
            assert graphs and generated > 0, (len(graphs), generated)
            path = args.output.parent / f"fx-{precision_id}.txt"
            path.write_text("\n\n".join(graphs))
            report["compilation"].append(
                dict(
                    precision=str(dtype),
                    graphs=len(graphs),
                    generated_kernels=generated,
                    fullgraph=True,
                    dynamic=False,
                )
            )
            save()
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
