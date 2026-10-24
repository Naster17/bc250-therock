# SPDX-License-Identifier: MIT
"""Experimental bounded W8 weight-only linear; explicitly imported, not installed.

Frozen symmetric per-output-channel INT8 weights, FP32 scales; GPU forward and
input/bias first derivatives. Not packed INT4/NF4, integer autograd, or a speed
claim. Native Torch operators and model dispatch are unchanged.
"""

from dataclasses import dataclass
import math

import torch
from torch.autograd.function import once_differentiable

MAX_WEIGHT_ELEMENTS = 1 << 22
MAX_ACTIVATION_ELEMENTS = 1 << 20
MAX_MACS = 1 << 27
MAX_DIMENSION = 4096
DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class FrozenInt8Weight:
    values: torch.Tensor
    scales: torch.Tensor
    version: int = 1


def _device(value: torch.Tensor) -> None:
    if value.device.type != "cuda" or not torch.version.hip:
        raise ValueError("The prototype requires native gfx1013 HIP tensors")
    if not torch.cuda.get_device_properties(value.device).gcnArchName.startswith(
        "gfx1013:"
    ):
        raise ValueError("The prototype is qualified only on native gfx1013")


def _weight_shape(shape) -> None:
    if len(shape) != 2 or not all(0 < d <= MAX_DIMENSION for d in shape):
        raise ValueError("Weight must be a nonempty 2D matrix with dimensions <=4096")
    if math.prod(shape) > MAX_WEIGHT_ELEMENTS:
        raise ValueError("Weight exceeds the prototype storage bound")


@torch.no_grad()
def pack(weight: torch.Tensor) -> FrozenInt8Weight:
    """Round-to-even/clamp[-127,127]; zero rows use scale1; retain no float base."""
    _weight_shape(weight.shape)
    _device(weight)
    if weight.dtype not in DTYPES or weight.requires_grad:
        raise ValueError("Pack an explicitly frozen FP16/BF16/FP32 weight")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Weight must be finite")
    floating = weight.float()
    maximum = floating.abs().amax(dim=1)
    # Multiplication specifies a single FP32 reciprocal constant for reproducible
    # scalar reference conversion; no row-size-dependent reduction of sums.
    scales = maximum * (1.0 / 127.0)
    scales = torch.where(maximum == 0, torch.ones_like(scales), scales)
    if not bool((scales > 0).all()):
        raise ValueError("Nonzero weight scale underflowed FP32")
    values = (floating / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    return FrozenInt8Weight(values.contiguous(), scales.contiguous())


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, values, scales):
        ctx.save_for_backward(values, scales)
        ctx.input_dtype = inputs.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            return torch._weight_int8pack_mm(inputs.contiguous(), values, scales)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        values, scales = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None
        with torch.autocast(device_type="cuda", enabled=False):
            # dX=(dY*row_scales)@integer_weight. Reuse the generic native kernel
            # with transposed INT8 storage, without materializing a float weight.
            scaled = (upstream.float() * scales).contiguous()
            transposed = values.T.contiguous()
            unit_scales = torch.ones(
                values.shape[1], device=values.device, dtype=torch.float32
            )
            result = torch._weight_int8pack_mm(scaled, transposed, unit_scales)
        return result.to(ctx.input_dtype), None, None


def linear(
    inputs: torch.Tensor, weight: FrozenInt8Weight, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Keep input dtype (including under AMP); add bias after the native cast.

    Arbitrary input views are normalized. Stored values/scales are frozen and
    validated; first derivatives only. Finite-data checks synchronize the host.
    The bounds concern operand sizes/work, not total process/allocator memory.
    """
    if not isinstance(weight, FrozenInt8Weight) or weight.version != 1:
        raise ValueError("Unknown frozen INT8 weight format")
    values, scales = weight.values, weight.scales
    _weight_shape(values.shape)
    _device(inputs)
    n, k = values.shape
    if inputs.ndim < 1 or inputs.shape[-1] != k or inputs.dtype not in DTYPES:
        raise ValueError("Input must have matching K and FP16/BF16/FP32 dtype")
    rows = math.prod(inputs.shape[:-1])
    if (
        rows < 1
        or rows > MAX_DIMENSION
        or inputs.numel() > MAX_ACTIVATION_ELEMENTS
        or rows * n > MAX_ACTIVATION_ELEMENTS
        or rows * n * k > MAX_MACS
    ):
        raise ValueError("Activation/output/work exceeds the bounded prototype")
    if (
        values.device != inputs.device
        or scales.device != inputs.device
        or values.dtype != torch.int8
        or scales.dtype != torch.float32
        or scales.shape != (n,)
        or scales.requires_grad
    ):
        raise ValueError(
            "Frozen INT8 values and FP32 row scales must share the input device"
        )
    scale_valid = (torch.isfinite(scales) & (scales > 0)).all()
    input_valid = torch.isfinite(inputs).all()
    if not bool(scale_valid & input_valid):
        if not bool(scale_valid):
            raise ValueError("Scales must be finite and positive")
        raise ValueError("Inputs must be finite")
    if bias is not None and (
        bias.shape != (n,) or bias.device != inputs.device or bias.dtype != inputs.dtype
    ):
        raise ValueError(
            "Bias must match output dimension, input device and input dtype"
        )
    output = _Linear.apply(
        inputs.reshape(rows, k), values.contiguous(), scales.contiguous()
    )
    output = output.reshape(*inputs.shape[:-1], n)
    return output if bias is None else output + bias
