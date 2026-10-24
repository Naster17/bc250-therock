#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned frozen-W8 module CPU references, strict state and optimizer resume."""

import argparse
import copy
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("module", "state", "resume"), required=True)
    args = parser.parse_args()
    import torch
    from bc250_quant_b055 import FrozenW8Linear
    from validate_quant_b051 import reference_pack

    torch.set_num_threads(1)
    torch.manual_seed(555)
    report = dict(status="running", checks=[], group=args.group)

    def passed(name):
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def exact(a, b):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)

    def state_cpu(state):
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().clone()
        if isinstance(state, dict):
            return {k: state_cpu(v) for k, v in state.items()}
        if isinstance(state, list):
            return [state_cpu(v) for v in state]
        if isinstance(state, tuple):
            return tuple(state_cpu(v) for v in state)
        return copy.deepcopy(state)

    def same_state(a, b):
        assert set(a) == set(b)
        for key in a:
            if isinstance(a[key], torch.Tensor):
                exact(a[key], b[key])
            else:
                assert a[key] == b[key], key

    def source(dtype, policy):
        cpu = torch.randn(7, 17).to(dtype)
        cpu[0].zero_()
        result = torch.nn.Linear(
            17, 7, bias=policy != "none", device="cuda", dtype=dtype
        )
        with torch.no_grad():
            result.weight.copy_(cpu)
            if result.bias is not None:
                result.bias.copy_(torch.arange(7, dtype=dtype) / 32)
        result.weight.requires_grad_(False)
        if result.bias is not None:
            result.bias.requires_grad_(policy == "trainable")
        result.eval()
        return result, cpu

    def refuses(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError, TypeError) as error:
            report.setdefault("refusals", []).append(dict(name=name, error=str(error)))
            passed("refuse-" + name)
        else:
            raise AssertionError("Missing refusal: " + name)

    try:
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
        if args.group == "module":
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                tol = (
                    dict(rtol=5e-5, atol=3e-6)
                    if dtype == torch.float32
                    else dict(rtol=3e-3, atol=5e-4)
                    if dtype == torch.float16
                    else dict(rtol=2e-2, atol=4e-3)
                )
                for policy in ("none", "frozen", "trainable"):
                    for layout in ("contiguous", "strided"):
                        src, w = source(dtype, policy)
                        before = src.weight.clone()
                        module = FrozenW8Linear.from_float(src)
                        q, s = reference_pack(w, torch)
                        exact(module.qweight, q)
                        exact(module.scales, s)
                        assert not module.training and not hasattr(module, "weight")
                        assert list(module.named_parameters()) == (
                            [] if policy == "none" else [("bias", module.bias)]
                        )
                        assert (
                            module.qweight.dtype == torch.int8
                            and module.scales.dtype == torch.float32
                        )
                        xr = (torch.randn(3, 17) / 4).to(dtype)
                        u = (torch.randn(3, 7) / 4).to(dtype)
                        if layout == "strided":
                            owner = torch.full(
                                (3, 37), 19.0, device="cuda", dtype=dtype
                            )
                            x = owner[:, 1:35:2]
                            x.copy_(xr)
                            sentinel = owner.clone()
                        else:
                            x = xr.cuda()
                        x.requires_grad_()
                        y = module(x)
                        y.backward(u.cuda())
                        base = (xr.double() @ (q.double() * s.double()[:, None]).T).to(
                            dtype
                        )
                        expected = (
                            base
                            if module.bias is None
                            else base + src.bias.detach().cpu()
                        )
                        torch.testing.assert_close(y.cpu(), expected, **tol)
                        torch.testing.assert_close(
                            x.grad.cpu(),
                            (u.double() @ (q.double() * s.double()[:, None])).to(dtype),
                            **tol,
                        )
                        if policy == "trainable":
                            torch.testing.assert_close(
                                module.bias.grad.cpu(),
                                u.float().sum(0).to(dtype),
                                **tol,
                            )
                        elif module.bias is not None:
                            assert module.bias.grad is None
                        if layout == "strided":
                            exact(owner, sentinel)
                        exact(src.weight, before)
                        assert src.weight.grad is None
                        if src.bias is not None:
                            assert (
                                src.bias.grad is None
                                and module.bias.data_ptr() != src.bias.data_ptr()
                            )
                        persistent = module.qweight.numel() + 4 * module.scales.numel()
                        assert persistent == 7 * 17 + 4 * 7
                        passed(
                            f"cpu-output-input-bias-grad-source-guards-{dtype}-{policy}-{layout}"
                        )
                src, _ = source(dtype, "trainable")
                module = FrozenW8Linear.from_float(src)
                q = module.qweight.clone()
                s = module.scales.clone()
                for target in (torch.float16, torch.bfloat16, torch.float32):
                    module.to(dtype=target)
                    assert (
                        module.compute_dtype == target and module.bias.dtype == target
                    )
                    assert module.scales.dtype == torch.float32
                    exact(module.qweight, q)
                    exact(module.scales, s)
                module.cpu()
                assert module.qweight.device.type == "cpu"
                same_state(
                    state_cpu(module.state_dict()),
                    state_cpu(module.cuda().cpu().state_dict()),
                )
                module.cuda()
                passed("dtype-device-moves-preserve-fp32-scales-" + str(dtype))
            module = FrozenW8Linear(3, 2, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                assert module(torch.ones(2, 3, device="cuda")).dtype == torch.float32
            passed("explicit-dtype-under-amp")
            original = state_cpu(module.state_dict())
            refuses("double-conversion", lambda: module.double())
            same_state(original, state_cpu(module.state_dict()))
            refuses("meta-conversion", lambda: module.to("meta"))
            same_state(original, state_cpu(module.state_dict()))
            refuses("cpu-forward", lambda: module.cpu()(torch.ones(2, 3)))
            module.cuda()
            refuses(
                "input-dtype",
                lambda: module(torch.ones(2, 3, device="cuda", dtype=torch.float16)),
            )
            refuses(
                "trainable-source",
                lambda: FrozenW8Linear.from_float(torch.nn.Linear(3, 2, device="cuda")),
            )
            cpu = torch.nn.Linear(3, 2)
            cpu.weight.requires_grad_(False)
            refuses("cpu-source", lambda: FrozenW8Linear.from_float(cpu))

            class Subclass(torch.nn.Linear):
                pass

            sub = Subclass(3, 2, device="cuda")
            sub.weight.requires_grad_(False)
            refuses("source-subclass", lambda: FrozenW8Linear.from_float(sub))
            refuses("oversized-constructor", lambda: FrozenW8Linear(4096, 4096))
            refuses("boolean-dim", lambda: FrozenW8Linear(True, 3))
            refuses(
                "bias-policy",
                lambda: FrozenW8Linear(3, 2, bias=False, bias_trainable=True),
            )
        elif args.group == "state":
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for policy in ("none", "frozen", "trainable"):
                    src, _ = source(dtype, policy)
                    module = FrozenW8Linear.from_float(src)
                    state = state_cpu(module.state_dict())
                    file = args.output.parent / f"state-{str(dtype)}-{policy}.pt"
                    torch.save(state, file)
                    loaded = torch.load(file, map_location="cpu", weights_only=True)
                    restored = FrozenW8Linear.from_state_dict(loaded).cuda()
                    same_state(state, state_cpu(restored.state_dict()))
                    x = torch.randn(3, 17, device="cuda", dtype=dtype)
                    exact(module(x), restored(x))
                    direct = FrozenW8Linear(
                        17,
                        7,
                        bias=policy != "none",
                        device="cuda",
                        dtype=dtype,
                        bias_trainable=policy == "trainable",
                    )
                    direct.load_state_dict(loaded)
                    exact(module(x), direct(x))
                    parent = torch.nn.Sequential(module)
                    nested = torch.nn.Sequential(
                        FrozenW8Linear.from_state_dict(loaded, device="cuda")
                    )
                    nested.load_state_dict(state_cpu(parent.state_dict()))
                    exact(parent(x), nested(x))
                    passed(
                        f"cpu-weights-only-file-factory-direct-nested-{dtype}-{policy}"
                    )
            src, _ = source(torch.float32, "trainable")
            module = FrozenW8Linear.from_float(src)
            base = state_cpu(module.state_dict())
            before = state_cpu(module.state_dict())
            bad = []

            def altered(name, fn):
                state = copy.deepcopy(base)
                state["qweight"].add_(1)
                fn(state)
                bad.append((name, state))

            altered(
                "version", lambda s: s["_extra_state"].__setitem__("format_version", 2)
            )
            altered(
                "boolean-version",
                lambda s: s["_extra_state"].__setitem__("format_version", True),
            )
            altered(
                "unknown-format",
                lambda s: s["_extra_state"].__setitem__("format", "foreign"),
            )
            altered(
                "metadata-missing", lambda s: s["_extra_state"].pop("compute_dtype")
            )
            altered(
                "metadata-unknown",
                lambda s: s["_extra_state"].__setitem__("unknown", 0),
            )
            altered(
                "metadata-dtype",
                lambda s: s["_extra_state"].__setitem__(
                    "compute_dtype", "torch.float64"
                ),
            )
            altered(
                "metadata-grad-policy",
                lambda s: s["_extra_state"].__setitem__("bias_trainable", False),
            )
            altered(
                "metadata-dim",
                lambda s: s["_extra_state"].__setitem__("in_features", 18),
            )
            altered("missing-values", lambda s: s.pop("qweight"))
            altered("missing-scales", lambda s: s.pop("scales"))
            altered("missing-version-state", lambda s: s.pop("_extra_state"))
            altered(
                "unknown-field", lambda s: s.__setitem__("weight", torch.ones(7, 17))
            )
            altered(
                "float-values", lambda s: s.__setitem__("qweight", s["qweight"].float())
            )
            altered(
                "wrong-values-shape",
                lambda s: s.__setitem__("qweight", s["qweight"][:, :-1]),
            )
            altered(
                "sparse-values",
                lambda s: s.__setitem__("qweight", s["qweight"].to_sparse()),
            )
            altered(
                "meta-values",
                lambda s: s.__setitem__(
                    "qweight", torch.empty((7, 17), device="meta", dtype=torch.int8)
                ),
            )
            altered(
                "low-precision-scales",
                lambda s: s.__setitem__("scales", s["scales"].half()),
            )
            altered(
                "scale-shape", lambda s: s.__setitem__("scales", s["scales"][:, None])
            )
            altered("zero-scale", lambda s: s["scales"].zero_())
            altered("negative-scale", lambda s: s["scales"].neg_())
            altered("nonfinite-scale", lambda s: s["scales"].fill_(float("nan")))
            altered("trainable-scale", lambda s: s["scales"].requires_grad_())
            altered("bias-dtype", lambda s: s.__setitem__("bias", s["bias"].half()))
            altered("bias-shape", lambda s: s.__setitem__("bias", s["bias"][:-1]))
            for name, state in bad:
                refuses("state-" + name, lambda: module.load_state_dict(state))
                same_state(before, state_cpu(module.state_dict()))
            partial = copy.deepcopy(base)
            partial.pop("scales")
            refuses(
                "partial-even-strict-false",
                lambda: module.load_state_dict(partial, strict=False),
            )
            same_state(before, state_cpu(module.state_dict()))
            refuses("assign-true", lambda: module.load_state_dict(base, assign=True))
            same_state(before, state_cpu(module.state_dict()))
            oversized = copy.deepcopy(base)
            oversized["_extra_state"].update(in_features=4096, out_features=4096)
            refuses(
                "factory-cap-before-allocation",
                lambda: FrozenW8Linear.from_state_dict(oversized),
            )
        else:
            src, w = source(torch.float32, "trainable")
            q, s = reference_pack(w, torch)
            wr = q.double() * s.double()[:, None]

            class Adapter(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.base = FrozenW8Linear.from_float(src)
                    self.a = torch.nn.Parameter(
                        torch.arange(34, device="cuda", dtype=torch.float32).reshape(
                            17, 2
                        )
                        / 256
                    )
                    self.b = torch.nn.Parameter(
                        torch.arange(14, device="cuda", dtype=torch.float32).reshape(
                            2, 7
                        )
                        / 256
                    )

                def forward(self, x):
                    return self.base(x) + (x @ self.a) @ self.b

            model = Adapter()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=0.01,
                foreach=False,
                fused=False,
            )
            reference = [
                model.a.detach().cpu().clone().requires_grad_(),
                model.b.detach().cpu().clone().requires_grad_(),
                model.base.bias.detach().cpu().clone().requires_grad_(),
            ]
            refopt = torch.optim.AdamW(
                reference, lr=0.001, weight_decay=0.01, foreach=False, fused=False
            )
            data = [
                (torch.arange(51).reshape(3, 17).float() / 64 + step / 16)
                for step in range(2)
            ]
            upstream = torch.arange(21).reshape(3, 7).float() / 128 - 0.0625

            def step(model, opt, index):
                opt.zero_grad()
                x = data[index].cuda().requires_grad_()
                y = model(x)
                y.backward(upstream.cuda())
                opt.step()
                return x, y

            for i in range(2):
                optimizer.zero_grad()
                refopt.zero_grad()
                x = data[i].cuda().requires_grad_()
                xr = data[i].clone().requires_grad_()
                y = model(x)
                expected = (
                    (xr.double() @ wr.T).float()
                    + reference[2]
                    + (xr @ reference[0]) @ reference[1]
                )
                torch.testing.assert_close(y.cpu(), expected, rtol=5e-5, atol=3e-6)
                y.backward(upstream.cuda())
                expected.backward(upstream)
                for actual, ref in zip(
                    (x, model.a, model.b, model.base.bias), (xr, *reference)
                ):
                    torch.testing.assert_close(
                        actual.grad.cpu(), ref.grad, rtol=5e-5, atol=3e-6
                    )
                optimizer.step()
                refopt.step()
                for actual, ref in zip((model.a, model.b, model.base.bias), reference):
                    torch.testing.assert_close(actual.cpu(), ref, rtol=5e-5, atol=3e-6)
                passed("cpu-all-first-gradients-and-adamw-step-" + str(i))
                if i == 0:
                    checkpoint = dict(
                        model=state_cpu(model.state_dict()),
                        optimizer=state_cpu(optimizer.state_dict()),
                        step=1,
                    )
                    file = args.output.parent / "owned-adapter-checkpoint.pt"
                    torch.save(checkpoint, file)
            loaded = torch.load(file, map_location="cpu", weights_only=True)
            assert loaded["step"] == 1
            resumed = Adapter()
            resumed.load_state_dict(loaded["model"])
            ropt = torch.optim.AdamW(
                resumed.parameters(),
                lr=0.001,
                weight_decay=0.01,
                foreach=False,
                fused=False,
            )
            ropt.load_state_dict(loaded["optimizer"])
            step(resumed, ropt, 1)
            same_state(state_cpu(model.state_dict()), state_cpu(resumed.state_dict()))
            for p1, p2 in zip(model.parameters(), resumed.parameters()):
                for key in optimizer.state[p1]:
                    a, b = optimizer.state[p1][key], ropt.state[p2][key]
                    if isinstance(a, torch.Tensor):
                        exact(a, b)
                    else:
                        assert a == b
            exact(model.base.qweight, q)
            exact(model.base.scales, s)
            assert model.base.qweight.grad is None and model.base.scales.grad is None
            passed("bitwise-continuous-vs-cpu-checkpoint-resume-model-optimizer")
        torch.cuda.synchronize()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
