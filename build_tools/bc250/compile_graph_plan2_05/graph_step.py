# SPDX-License-Identifier: MIT
"""Reusable opt-in HIP graph-capture helper for a training step (PLAN2-05).

This is the qualified recipe artifact: capture a zero-argument step closure that
operates on STATIC tensors (model parameters, static input/target buffers, and
optionally a capturable optimizer) and replay it with one launch. It changes no
default and installs nothing; callers import it deliberately.

Boundaries (qualified in PLAN2-05): dropout must be 0 (RNG is not graph-safe);
input shapes must be static (a new shape needs recapture); the graph memory pool
is not checkpointable, so checkpoint/resume requires a fresh recapture from
loaded state. Keep the eager autograd graph from leaking into the capture (drop
eager outputs and zero_grad before capturing) or stale AccumulateGrad stream
references invalidate it.
"""

from __future__ import annotations

import torch


class GraphCaptureError(RuntimeError):
    """Raised when a step is not graph-safe (dropout / dynamic shape)."""


def assert_graph_safe(model, static_inputs):
    for module in model.modules():
        if isinstance(
            module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d)
        ):
            if getattr(module, "p", 0.0) > 0.0:
                raise GraphCaptureError("dropout > 0 is not graph-safe (RNG)")
    for tensor in static_inputs:
        if tensor.requires_grad and tensor.grad is not None:
            # A lingering eager graph on a static input invalidates capture.
            raise GraphCaptureError("static input carries a stale eager graph")


class GraphedStep:
    """Capture and replay a zero-argument training step on static tensors.

    step_fn must perform the whole step (e.g. zero_grad + forward + backward,
    and optionally a capturable optimizer.step()) using only static tensors, and
    return the loss tensor (or None). The loss is read from a static buffer the
    closure writes, so replay updates it in place.
    """

    def __init__(self, step_fn, warmup: int = 3):
        self.step_fn = step_fn
        self.warmup = warmup
        self.graph = None

    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self.step_fn()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step_fn()
        return self

    def replay(self):
        if self.graph is None:
            raise GraphCaptureError("capture() before replay()")
        self.graph.replay()

    def reset(self):
        self.graph = None


def capture_graphed_callable(model, sample_args, **kwargs):
    """Thin wrapper over torch.cuda.make_graphed_callables for fwd+bwd capture.

    The caller still runs the optimizer eagerly; this graphs the module's
    forward+backward only. Drop any eager output graph before calling.
    """
    return torch.cuda.make_graphed_callables(model, sample_args, **kwargs)
