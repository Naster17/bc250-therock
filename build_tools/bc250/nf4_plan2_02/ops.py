# SPDX-License-Identifier: MIT
"""Native NF4 weight-only linear for gfx1013 (torch; explicit opt-in).

Frozen 4-bit NormalFloat weights with per-block absmax FP32 scales. Forward
dequantizes to the input compute dtype and uses the qualified rocBLAS GEMM; the
input first derivative recomputes the dequantized weight from the compact saved
codes/scales (no full floating base is persisted). RDNA1 has no INT4 matrix
cores, so this is a memory win, not a compute win. Native operators, model
dispatch and defaults are unchanged; packed native INT4 stays hardware-refused.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from . import core

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DTYPE_NAMES = {str(dtype): dtype for dtype in DTYPES}


def _device(value: torch.Tensor) -> None:
    """Storage/pack guard: CPU is allowed for storage/reference; CUDA must be gfx1013."""
    if value.device.type == "cpu":
        return
    if value.device.type != "cuda" or not torch.version.hip:
        raise ValueError("NF4 requires native gfx1013 HIP tensors (or CPU storage)")
    if not torch.cuda.get_device_properties(value.device).gcnArchName.startswith(
        "gfx1013:"
    ):
        raise ValueError("NF4 is qualified only on native gfx1013")


def _compute_device(value: torch.Tensor) -> None:
    """Compute guard: the qualified linear forward/backward path is gfx1013-only."""
    if value.device.type != "cuda" or not torch.version.hip:
        raise ValueError("NF4 linear requires native gfx1013 HIP tensors")
    if not torch.cuda.get_device_properties(value.device).gcnArchName.startswith(
        "gfx1013:"
    ):
        raise ValueError("NF4 linear is qualified only on native gfx1013")


def _codebook(device) -> torch.Tensor:
    return torch.tensor(core.NF4_CODEBOOK, device=device, dtype=torch.float32)


def _boundaries(device) -> torch.Tensor:
    return torch.tensor(core.NF4_MIDPOINTS, device=device, dtype=torch.float32)


@dataclass(frozen=True)
class FrozenNF4Weight:
    codes: torch.Tensor  # uint8 [n, k//2], two 4-bit codes per byte
    scales: torch.Tensor  # float32 [n, k//block_size]
    block_size: int = core.DEFAULT_BLOCK_SIZE
    version: int = 1


@torch.no_grad()
def pack(
    weight: torch.Tensor, block_size: int = core.DEFAULT_BLOCK_SIZE
) -> FrozenNF4Weight:
    """Block-wise absmax NF4 pack of an explicitly frozen FP16/BF16/FP32 weight.

    Retains no floating base. Zero blocks use scale 1; nonzero scale underflow is
    refused. in_features must be divisible by the block size.
    """
    core.validate_weight_shape(weight.shape)
    core.validate_block_size(block_size)
    _device(weight)
    n, k = weight.shape
    if k % block_size != 0:
        raise ValueError("in_features must be divisible by the NF4 block size")
    if weight.dtype not in DTYPES or weight.requires_grad:
        raise ValueError("Pack an explicitly frozen FP16/BF16/FP32 weight")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Weight must be finite")
    floating = weight.float().reshape(n, k // block_size, block_size)
    maximum = floating.abs().amax(dim=-1)
    scales = torch.where(maximum == 0, torch.ones_like(maximum), maximum)
    if not bool((scales > 0).all()):
        raise ValueError("Nonzero weight scale underflowed FP32")
    normalized = (floating / scales.unsqueeze(-1)).reshape(-1)
    # Nearest level via the sorted midpoints; ties round to the higher code,
    # matching core.nearest_code (proved by the local reference tests).
    indices = torch.searchsorted(_boundaries(weight.device), normalized, right=True)
    indices = indices.reshape(n, k).to(torch.int64)
    low = indices[:, 0::2]
    high = indices[:, 1::2]
    codes = (low | (high << 4)).to(torch.uint8).contiguous()
    return FrozenNF4Weight(codes, scales.contiguous(), block_size)


def dequantize(weight: FrozenNF4Weight, dtype: torch.dtype) -> torch.Tensor:
    """Restore the [n, k] weight in `dtype` from compact codes/scales."""
    if not isinstance(weight, FrozenNF4Weight) or weight.version != 1:
        raise ValueError("Unknown frozen NF4 weight format")
    core.validate_block_size(weight.block_size)
    codes, scales = weight.codes, weight.scales
    n, k2 = codes.shape
    k = k2 * 2
    core.validate_weight_shape((n, k))
    if k % weight.block_size != 0:
        raise ValueError("in_features must be divisible by the NF4 block size")
    if codes.dtype != torch.uint8 or scales.dtype != torch.float32:
        raise ValueError("NF4 codes must be uint8 and scales float32")
    if scales.shape != (n, k // weight.block_size):
        raise ValueError("NF4 scale shape does not match the block layout")
    # Memory-lean dequant: gather the two nibbles separately (half-size int64
    # indices) and apply scales by blocked broadcast rather than a full
    # repeat_interleave. The element-wise FP32 codebook*scale product is
    # bit-identical to the dense form, so the parity contract is unchanged.
    cb = _codebook(codes.device)
    low = (codes & 0xF).to(torch.int64)
    high = (codes >> 4).to(torch.int64)
    normalized = torch.stack((cb[low], cb[high]), dim=-1).reshape(n, k)
    bs = weight.block_size
    blocked = normalized.reshape(n, k // bs, bs) * scales.unsqueeze(-1)
    return blocked.reshape(n, k).to(dtype)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, codes, scales, block_size):
        ctx.save_for_backward(codes, scales)
        ctx.block_size = block_size
        ctx.input_dtype = inputs.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            weight = dequantize(
                FrozenNF4Weight(codes, scales, block_size), inputs.dtype
            )
            return torch.matmul(inputs, weight.t())

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        codes, scales = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        with torch.autocast(device_type="cuda", enabled=False):
            weight = dequantize(
                FrozenNF4Weight(codes, scales, ctx.block_size), torch.float32
            )
            result = torch.matmul(upstream.float(), weight)
        return result.to(ctx.input_dtype), None, None, None


def linear(
    inputs: torch.Tensor,
    weight: FrozenNF4Weight,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep input dtype (including under AMP); add bias after the GEMM.

    First input derivative only; the base weight/scales are frozen. Finite-data
    checks synchronize the host. Bounds concern operand sizes/work per call.
    """
    if not isinstance(weight, FrozenNF4Weight) or weight.version != 1:
        raise ValueError("Unknown frozen NF4 weight format")
    core.validate_block_size(weight.block_size)
    codes, scales = weight.codes, weight.scales
    n, k = codes.shape[0], codes.shape[1] * 2
    core.validate_weight_shape((n, k))
    if k % weight.block_size != 0:
        raise ValueError("in_features must be divisible by the NF4 block size")
    _compute_device(inputs)
    if inputs.ndim < 1 or inputs.shape[-1] != k or inputs.dtype not in DTYPES:
        raise ValueError("Input must have matching K and FP16/BF16/FP32 dtype")
    rows = math.prod(inputs.shape[:-1])
    if (
        rows < 1
        or rows > core.MAX_DIMENSION
        or inputs.numel() > core.MAX_ACTIVATION_ELEMENTS
        or rows * n > core.MAX_ACTIVATION_ELEMENTS
        or rows * n * k > core.MAX_MACS
    ):
        raise ValueError("Activation/output/work exceeds the bounded prototype")
    if (
        codes.device != inputs.device
        or scales.device != inputs.device
        or codes.dtype != torch.uint8
        or scales.dtype != torch.float32
        or scales.shape != (n, k // weight.block_size)
        or scales.requires_grad
    ):
        raise ValueError("Frozen NF4 codes and FP32 block scales must share the device")
    scale_valid = (torch.isfinite(scales) & (scales > 0)).all()
    input_valid = torch.isfinite(inputs).all()
    if not bool(scale_valid & input_valid):
        if not bool(scale_valid):
            raise ValueError("Scales must be finite and positive")
        raise ValueError("Inputs must be finite")
    if bias is not None and (
        bias.shape != (n,) or bias.device != inputs.device or bias.dtype != inputs.dtype
    ):
        raise ValueError("Bias must match output dimension, input device and dtype")
    output = _Linear.apply(inputs.reshape(rows, k), codes, scales, weight.block_size)
    output = output.reshape(*inputs.shape[:-1], n)
    return output if bias is None else output + bias


class FrozenNF4Linear(nn.Module):
    """Frozen NF4 buffers and optional trainable bias; eager native gfx1013 only.

    CPU storage/load is allowed; forward and float conversion require gfx1013.
    State format v1 is complete/strict even with load_state_dict(strict=False).
    Convert before constructing an optimizer. No automatic model replacement.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        *,
        block_size=core.DEFAULT_BLOCK_SIZE,
        device="cpu",
        dtype=torch.float32,
        bias_trainable=False,
    ):
        super().__init__()
        if type(in_features) is not int or type(out_features) is not int:
            raise ValueError("Feature dimensions must be integers")
        core.validate_weight_shape((out_features, in_features))
        core.validate_block_size(block_size)
        if in_features % block_size != 0:
            raise ValueError("in_features must be divisible by the NF4 block size")
        if (
            dtype not in DTYPES
            or type(bias) is not bool
            or type(bias_trainable) is not bool
        ):
            raise ValueError("Unsupported compute dtype or bias policy")
        if not bias and bias_trainable:
            raise ValueError("A missing bias cannot be trainable")
        probe = torch.empty(0, device=device, dtype=dtype)
        self._target(probe)
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, in_features // 2), device=probe.device, dtype=torch.uint8
            ),
        )
        self.register_buffer(
            "scales",
            torch.ones(
                (out_features, in_features // block_size),
                device=probe.device,
                dtype=torch.float32,
            ),
        )
        self.register_buffer("_dtype_anchor", probe, persistent=False)
        self.register_parameter(
            "bias",
            (
                nn.Parameter(
                    torch.zeros(out_features, device=probe.device, dtype=dtype),
                    requires_grad=bias_trainable,
                )
                if bias
                else None
            ),
        )

    @staticmethod
    def _target(probe):
        if probe.dtype not in DTYPES or probe.device.type not in ("cpu", "cuda"):
            raise ValueError(
                "Only CPU storage/native gfx1013 and FP16/BF16/FP32 are supported"
            )
        if probe.device.type == "cuda":
            _device(probe)

    @property
    def compute_dtype(self):
        return self._dtype_anchor.dtype

    @classmethod
    @torch.no_grad()
    def from_float(cls, source, block_size=core.DEFAULT_BLOCK_SIZE):
        if type(source) is not nn.Linear:
            raise ValueError("Convert an explicit standard nn.Linear, not a subclass")
        packed = pack(source.weight, block_size)
        result = cls(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            block_size=block_size,
            device=source.weight.device,
            dtype=source.weight.dtype,
            bias_trainable=source.bias is not None and source.bias.requires_grad,
        )
        result.qweight = packed.codes
        result.scales = packed.scales
        if source.bias is not None:
            if (
                source.bias.device != source.weight.device
                or source.bias.dtype != source.weight.dtype
            ):
                raise ValueError(
                    "Source bias must match source weight device and dtype"
                )
            result.bias.copy_(source.bias)
        result.train(source.training)
        return result

    def forward(self, inputs):
        if inputs.dtype != self.compute_dtype:
            raise ValueError("Input dtype must match the module compute dtype")
        if self.qweight.shape != (
            self.out_features,
            self.in_features // 2,
        ) or self.scales.shape != (
            self.out_features,
            self.in_features // self.block_size,
        ):
            raise ValueError("Module payload shape changed")
        return linear(
            inputs,
            FrozenNF4Weight(self.qweight, self.scales, self.block_size),
            self.bias,
        )

    def _apply(self, fn, recurse=True):
        old_anchor, old_scales = self._dtype_anchor, self.scales
        anchor = fn(old_anchor)
        self._target(anchor)
        scales = old_scales.to(device=anchor.device, dtype=torch.float32)

        def transform(value):
            if value is old_anchor:
                return anchor
            if value is old_scales:
                return scales
            return fn(value)

        return super()._apply(transform, recurse=recurse)

    def get_extra_state(self):
        return dict(
            format_version=core.FORMAT_VERSION,
            format=core.FORMAT_NAME,
            in_features=self.in_features,
            out_features=self.out_features,
            block_size=self.block_size,
            compute_dtype=str(self.compute_dtype),
            bias=self.bias is not None,
            bias_trainable=self.bias is not None and self.bias.requires_grad,
        )

    def set_extra_state(self, state):
        if core.validate_metadata(state) != self.get_extra_state():
            raise ValueError(
                "Frozen-NF4 metadata does not match the destination module"
            )

    def _payload(self, state, prefix):
        self.set_extra_state(state.get(prefix + "_extra_state"))
        required = {prefix + "qweight", prefix + "scales", prefix + "_extra_state"}
        if self.bias is not None:
            required.add(prefix + "bias")
        if {k for k in state if k.startswith(prefix)} != required:
            raise ValueError(
                "Frozen-NF4 payload must contain exactly the versioned fields"
            )
        values, scales = state[prefix + "qweight"], state[prefix + "scales"]
        if (
            not isinstance(values, torch.Tensor)
            or values.layout != torch.strided
            or values.device.type not in ("cpu", "cuda")
            or values.dtype != torch.uint8
            or values.shape != (self.out_features, self.in_features // 2)
        ):
            raise ValueError("Invalid NF4 code payload")
        if (
            not isinstance(scales, torch.Tensor)
            or scales.layout != torch.strided
            or scales.device != values.device
            or scales.dtype != torch.float32
            or scales.shape != (self.out_features, self.in_features // self.block_size)
            or scales.requires_grad
        ):
            raise ValueError("Invalid FP32 block-scale payload")
        if not bool((torch.isfinite(scales) & (scales > 0)).all()):
            raise ValueError("Scale payload must be finite and positive")
        if self.bias is not None:
            bias = state[prefix + "bias"]
            if (
                not isinstance(bias, torch.Tensor)
                or bias.layout != torch.strided
                or bias.device != values.device
                or bias.dtype != self.compute_dtype
                or bias.shape != (self.out_features,)
            ):
                raise ValueError("Invalid bias payload")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        try:
            if local_metadata.get("assign_to_params_buffers", False):
                raise ValueError("Frozen-NF4 loading requires assign=False")
            self._payload(state_dict, prefix)
        except (ValueError, TypeError, KeyError) as error:
            error_msgs.append(f"{prefix}frozen-NF4 state refused: {error}")
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_state_dict(cls, state_dict, *, device="cpu"):
        if not isinstance(state_dict, Mapping):
            raise ValueError("Expected a frozen-NF4 state mapping")
        meta = core.validate_metadata(state_dict.get("_extra_state"))
        result = cls(
            meta["in_features"],
            meta["out_features"],
            bias=meta["bias"],
            block_size=meta["block_size"],
            device=device,
            dtype=DTYPE_NAMES[meta["compute_dtype"]],
            bias_trainable=meta["bias_trainable"],
        )
        result.load_state_dict(state_dict)
        return result

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, dtype={self.compute_dtype}, "
            f"bias={self.bias is not None}"
        )


class QLoRALinear(nn.Module):
    """Frozen NF4 base + trainable FP16/BF16/FP32 low-rank adapter (QLoRA).

    y = NF4_base(x) + (x @ A^T) @ B^T * (alpha / rank). The base weight/scales
    stay frozen; only the adapter (and optional bias) carry gradients. This is
    the owned qualification composite, not an automatic model conversion.
    """

    def __init__(self, base: FrozenNF4Linear, rank: int, alpha: float):
        super().__init__()
        if type(rank) is not int or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if base.bias is not None and base.bias.requires_grad:
            raise ValueError("QLoRA composite expects a frozen base bias")
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        dtype = base.compute_dtype
        device = base.qweight.device
        self.lora_a = nn.Parameter(
            torch.zeros(rank, base.in_features, device=device, dtype=dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        base_out = self.base(inputs)
        adapter = torch.matmul(torch.matmul(inputs, self.lora_a.t()), self.lora_b.t())
        return base_out + adapter * self.scaling
