#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit FP32 evaluation scope for a selected single-input FP32-master head."""
from contextlib import contextmanager
from functools import wraps

import torch


def _validate(head):
    if not isinstance(head, torch.nn.Module) or any(m.training for m in head.modules()):
        raise ValueError("Select an existing head with all modules in eval mode")
    tensors = list(head.parameters()) + list(head.buffers())
    if not tensors or any(t.device != tensors[0].device for t in tensors):
        raise ValueError("Head tensors must reside on one CPU or CUDA device")
    if tensors[0].device.type not in ["cpu", "cuda"] or any(
        t.is_floating_point() and t.dtype != torch.float32 for t in tensors
    ):
        raise ValueError("Head must already use FP32 floating weights and buffers")
    return tensors[0].device


@contextmanager
def fp32_eval_head(head):
    """Disable autocast for ONE existing head and cast its hidden input to FP32.

    Caller opts in explicitly, using eval mode and no_grad/inference_mode. Head
    forward must accept one positional floating hidden-state tensor and return
    an FP32 tensor. Parameters/buffers/keys/ties are unchanged; scope restores the
    instance forward even on exceptions. No training, automatic model selection,
    compilation/concurrent same-instance calls or mutation during scope support.
    BF16 encoder arithmetic is retained and can still differ from FP64.
    """
    device = _validate(head)
    if getattr(head, "_bc250_fp32_eval_scope", False):
        raise ValueError("Head already has an active FP32 evaluation scope")
    had_forward = "forward" in head.__dict__
    instance_forward = head.__dict__.get("forward")
    original = head.forward

    @wraps(original)
    def forward(*args, **kwargs):
        if torch.is_grad_enabled() or _validate(head) != device:
            raise ValueError(
                "FP32 head scope requires no_grad/inference_mode and fixed device"
            )
        if (
            len(args) != 1
            or kwargs
            or not isinstance(args[0], torch.Tensor)
            or not args[0].is_floating_point()
            or args[0].device != device
        ):
            raise ValueError("Expected one same-device floating hidden-state tensor")
        with torch.autocast(device_type=device.type, enabled=False):
            result = original(args[0].float())
        if not isinstance(result, torch.Tensor) or result.dtype != torch.float32:
            raise ValueError("Selected head must return one FP32 tensor")
        return result

    head._bc250_fp32_eval_scope = True
    head.forward = forward
    try:
        yield head
    finally:
        if had_forward:
            head.forward = instance_forward
        else:
            del head.forward
        del head._bc250_fp32_eval_scope
