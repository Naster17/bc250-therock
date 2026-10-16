#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded native first-order training API references; synthetic tensors only."""
import json
import io
import copy
import os
import sys
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests
from torch.utils.checkpoint import checkpoint

if os.environ.get("BC250_A05_INSTALLED"):
    from torch.nn import bc250_training_ops as ops
else:
    import bc250_training_ops as ops

ROWS = []


def record(name, **data):
    ROWS.append(dict(name=name, status="passed", module=ops.__file__,
                     source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(), **data))
    Path(os.environ["BC250_A05_ROWS"]).write_text(json.dumps(ROWS, indent=2) + "\n")


def reference(x, w, b, activation):
    z = F.conv1d(x.double(), w.double().unsqueeze(1),
                 None if b is None else b.double(), padding=w.shape[1] - 1,
                 groups=x.shape[1])[..., :x.shape[-1]].to(x.dtype)
    return F.silu(z.double()).to(x.dtype) if activation else z


class TestTrainingA05(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(250505)
        torch.set_num_threads(2)
        self.assertEqual(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx1013")

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("shape", [(1, 3, 1, 1), (2, 7, 7, 2), (1, 9, 33, 3), (2, 5, 1031, 4)])
    @parametrize("activation", [None, "silu"])
    @parametrize("strided", [False, True])
    def test_conv(self, device, dtype, shape, activation, strided):
        n, c, length, k = shape
        storage = (torch.randn(n, c, length * 2, device=device, dtype=dtype) * .2)
        x = storage[..., ::2] if strided else storage[..., :length].contiguous()
        w = (torch.randn(c, k * 2, device=device, dtype=dtype) * .2)[:, ::2]
        b = (torch.randn(c * 2, device=device, dtype=dtype) * .1)[::2]
        args = [t.detach().requires_grad_() for t in (x, w, b)]
        saved = [t.detach().clone() for t in args]
        cpu = [t.detach().cpu().requires_grad_() for t in args]
        g = torch.randn(n, c, length, device=device, dtype=dtype) * .2
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.assertTrue(ops.supports_causal_conv(*args, activation))
            actual = ops.causal_conv(*args, activation)
            grads = torch.autograd.grad(actual, args, g)
        torch.cuda.current_stream().wait_stream(stream)
        expected = reference(*cpu, activation)
        refs = torch.autograd.grad(expected, cpu, g.cpu())
        tol = {torch.float32: (3e-6, 4e-5), torch.float16: (.003, .006), torch.bfloat16: (.025, .04)}[dtype]
        self.assertEqual(actual.cpu(), expected, atol=tol[0], rtol=tol[1])
        for result, ref in zip(grads, refs):
            self.assertEqual(result.cpu(), ref, atol=tol[0], rtol=tol[1])
        for result, before in zip(args, saved):
            self.assertEqual(result.detach(), before, atol=0, rtol=0)
        record(self._testMethodName, shape=shape, all_gradients=True, custom_stream=True)

    @parametrize("amp_dtype", [torch.float16, torch.bfloat16])
    @parametrize("recompute", [False, True])
    def test_amp_checkpoint_update(self, device, amp_dtype, recompute):
        args = [torch.randn(2, 5, 37, device=device) * .1,
                torch.randn(5, 4, device=device) * .2,
                torch.randn(5, device=device) * .1]
        args = [t.requires_grad_() for t in args]
        cpu = [t.detach().cpu().to(amp_dtype).requires_grad_() for t in args]
        def forward(*tensors):
            return ops.causal_conv(*tensors, activation="silu")
        with torch.autocast("cuda", dtype=amp_dtype):
            y = checkpoint(forward, *args, use_reentrant=False) if recompute else forward(*args)
        self.assertEqual(y.dtype, amp_dtype)
        expected = reference(*cpu, "silu")
        g = torch.randn_like(expected) * .1
        y.backward(g.to(device) * 8)
        expected.backward(g * 8)
        for a, r in zip(args, cpu):
            self.assertEqual(a.grad.cpu(), r.grad.float(), atol=.025 if amp_dtype == torch.bfloat16 else .003, rtol=.04)
        # Scaled finite gradient + one optimizer update; no broad AMP claim.
        before = args[1].detach().clone()
        for a in args:
            a.grad.div_(8)
        optimizer = torch.optim.SGD(args, lr=.01)
        expected_weight = before - .01 * args[1].grad
        optimizer.step()
        self.assertEqual(args[1], expected_weight)
        record(self._testMethodName, checkpoint=recompute, amp=str(amp_dtype))

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("mode", ["mean", "count", "shift", "ignored"])
    def test_loss(self, device, dtype, mode):
        x = torch.randn(2, 13, 29, device=device, dtype=dtype, requires_grad=True)
        labels = torch.randint(29, (2, 13), device=device)
        labels[:, 3] = -100
        if mode == "ignored":
            labels.fill_(-100)
        target = labels if mode == "shift" else F.pad(labels, (0, 1), value=-100)[..., 1:]
        kw = dict(shift_labels=target) if mode == "shift" else {}
        if mode == "count":
            kw["num_items_in_batch"] = 37
        loss = ops.chunked_causal_loss(x, labels, chunk_tokens=7, **kw)
        grad, = torch.autograd.grad(loss * 3, x)
        cpu = x.detach().cpu().double().requires_grad_()
        expected = F.cross_entropy(cpu.reshape(-1, 29), target.cpu().reshape(-1), reduction="sum" if mode == "count" else "mean")
        if mode == "count":
            expected = expected / 37
        ref, = torch.autograd.grad(expected * 3, cpu)
        if mode == "ignored":
            self.assertTrue(bool(torch.isnan(loss)))
        else:
            self.assertEqual(loss.cpu().double(), expected, atol=3e-6, rtol=3e-6)
        self.assertEqual(grad.cpu().double(), ref, atol=.003 if dtype == torch.bfloat16 else .0001, rtol=.04)
        record(self._testMethodName)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    def test_no_bias(self, device, dtype):
        for activation in [None, "silu"]:
            x = torch.randn(1, 7, 65, device=device, dtype=dtype).requires_grad_()
            w = (torch.randn(7, 4, device=device, dtype=dtype) * .1).requires_grad_()
            cpu = [a.detach().cpu().requires_grad_() for a in [x, w]]
            y = ops.causal_conv(x, w, activation=activation)
            ref = reference(*cpu, None, activation)
            grads = torch.autograd.grad(y.sum(), [x, w])
            refs = torch.autograd.grad(ref.sum(), cpu)
            tol = .06 if dtype == torch.bfloat16 else .006 if dtype == torch.float16 else 3e-5
            self.assertEqual(y.cpu(), ref, atol=tol, rtol=tol)
            for a, b in zip(grads, refs):
                self.assertEqual(a.cpu(), b, atol=tol, rtol=tol)
        record(self._testMethodName, no_bias=True)

    def test_model_resume(self, device):
        parameters = [torch.nn.Parameter(torch.randn(5, 4, device=device) * .1),
                      torch.nn.Parameter(torch.randn(5, device=device) * .1),
                      torch.nn.Parameter(torch.randn(5, 17, device=device) * .1)]
        optimizer = torch.optim.AdamW(parameters, lr=.003)
        labels = torch.arange(11, device=device)[None, :] % 17
        def step(check_cpu=False):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(1, 5, 11, device=device) * .2
            y = checkpoint(lambda a, b: ops.causal_conv(x, a, b, "silu"),
                           *parameters[:2], use_reentrant=False)
            logits = (y.transpose(1, 2) @ parameters[2]).contiguous()
            loss = ops.chunked_causal_loss(logits, labels, chunk_tokens=3)
            loss.backward()
            if check_cpu:
                cpu = [a.detach().cpu().requires_grad_() for a in parameters]
                ref_y = reference(x.cpu(), *cpu[:2], "silu")
                ref_logits = ref_y.transpose(1, 2) @ cpu[2]
                target = F.pad(labels.cpu(), (0, 1), value=-100)[..., 1:]
                ref_loss = F.cross_entropy(ref_logits.reshape(-1, 17), target.reshape(-1))
                ref_loss.backward()
                self.assertEqual(loss.cpu(), ref_loss, atol=2e-6, rtol=2e-5)
                for a, b in zip(parameters, cpu):
                    self.assertEqual(a.grad.cpu(), b.grad, atol=2e-6, rtol=3e-5)
            optimizer.step()
        step(check_cpu=True)
        state = dict(parameters=[a.detach().clone() for a in parameters],
                     optimizer=copy.deepcopy(optimizer.state_dict()),
                     gpu_rng=torch.cuda.get_rng_state())
        buffer = io.BytesIO()
        torch.save(state, buffer)
        step()
        expected = [a.detach().clone() for a in parameters]
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        with torch.no_grad():
            for a, b in zip(parameters, loaded["parameters"]):
                a.copy_(b)
        optimizer.load_state_dict(loaded["optimizer"])
        torch.cuda.set_rng_state(loaded["gpu_rng"])
        step()
        for a, b in zip(parameters, expected):
            self.assertEqual(a, b, atol=0, rtol=0)
        self.assertEqual(optimizer.state_dict(), expected_optimizer, atol=0, rtol=0)
        record(self._testMethodName, exact_resume=True, cpu_loss_all_parameter_gradients=True)

    def test_dispatch_and_guards(self, device):
        for k in [1, 5]:
            for flags in [(True, False, False), (False, True, False), (False, False, True)]:
                args = [torch.randn(1, 3, 13, device=device), torch.randn(3, k, device=device), torch.randn(3, device=device)]
                args = [a.requires_grad_(f) for a, f in zip(args, flags)]
                self.assertEqual(ops.supports_causal_conv(*args, None), k == 1)
                y = ops.causal_conv(*args)
                cpu = [a.detach().cpu().requires_grad_(f) for a, f in zip(args, flags)]
                ref = reference(*cpu, None)
                y.sum().backward(); ref.sum().backward()
                self.assertEqual(y.cpu(), ref, atol=3e-6, rtol=3e-5)
                for a, r in zip(args, cpu):
                    if a.requires_grad:
                        self.assertEqual(a.grad.cpu(), r.grad, atol=3e-6, rtol=3e-5)
        # CPU fallback and chunked loss remain callable without model hooks.
        a, w = torch.randn(1, 2, 7), torch.randn(2, 4)
        self.assertEqual(ops.causal_conv(a, w), F.conv1d(a, w[:, None], padding=3, groups=2)[..., :7])
        logits = torch.randn(1, 3, 5, device=device, requires_grad=True)
        labels = torch.zeros(1, 3, device=device, dtype=torch.long)
        for kwargs in [dict(chunk_tokens=0), dict(ignore_index=0), dict(num_items_in_batch=float("nan")), dict(num_items_in_batch=-1), dict(vocab_size=6)]:
            with self.assertRaises(ValueError):
                ops.chunked_causal_loss(logits, labels, **kwargs)
        with self.assertRaises(ValueError):
            ops.chunked_causal_loss(logits, labels + 5)
        with self.assertRaises(TypeError):
            ops.chunked_causal_loss(logits, labels, label_smoothing=.1)
        with self.assertRaises(ValueError):
            ops.causal_conv(a, w, activation="relu")
        record(self._testMethodName, independent_gradient_dispatches=6, refusals=8)


instantiate_device_type_tests(TestTrainingA05, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
