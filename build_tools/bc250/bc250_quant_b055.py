# SPDX-License-Identifier: MIT
"""Explicit bounded frozen-W8 module and versioned tensor-only state payload."""

from collections.abc import Mapping

import torch
from torch import nn

import bc250_quant_b051 as quant

DTYPE_NAMES = {str(dtype): dtype for dtype in quant.DTYPES}


class FrozenW8Linear(nn.Module):
    """Frozen INT8 buffers and optional trainable bias; eager native gfx1013 only.

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
        device="cpu",
        dtype=torch.float32,
        bias_trainable=False,
    ):
        super().__init__()
        if type(in_features) is not int or type(out_features) is not int:
            raise ValueError("Feature dimensions must be integers")
        quant._weight_shape((out_features, in_features))
        if (
            dtype not in quant.DTYPES
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
        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, in_features), device=probe.device, dtype=torch.int8
            ),
        )
        self.register_buffer(
            "scales", torch.ones(out_features, device=probe.device, dtype=torch.float32)
        )
        self.register_buffer("_dtype_anchor", probe, persistent=False)
        self.register_parameter(
            "bias",
            nn.Parameter(
                torch.zeros(out_features, device=probe.device, dtype=dtype),
                requires_grad=bias_trainable,
            )
            if bias
            else None,
        )

    @staticmethod
    def _target(probe):
        if probe.dtype not in quant.DTYPES or probe.device.type not in ("cpu", "cuda"):
            raise ValueError(
                "Only CPU storage/native gfx1013 and FP16/BF16/FP32 are supported"
            )
        if probe.device.type == "cuda":
            quant._device(probe)

    @property
    def compute_dtype(self):
        return self._dtype_anchor.dtype

    @classmethod
    @torch.no_grad()
    def from_float(cls, source):
        if type(source) is not nn.Linear:
            raise ValueError("Convert an explicit standard nn.Linear, not a subclass")
        packed = quant.pack(source.weight)
        result = cls(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
            bias_trainable=source.bias is not None and source.bias.requires_grad,
        )
        result.qweight = packed.values
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
            self.in_features,
        ) or self.scales.shape != (self.out_features,):
            raise ValueError("Module payload shape changed")
        return quant.linear(
            inputs, quant.FrozenInt8Weight(self.qweight, self.scales), self.bias
        )

    def _apply(self, fn, recurse=True):
        # Do not round scale buffers through the requested floating compute dtype.
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
            format_version=1,
            format="signed_int8_per_output_scale",
            in_features=self.in_features,
            out_features=self.out_features,
            compute_dtype=str(self.compute_dtype),
            bias=self.bias is not None,
            bias_trainable=self.bias is not None and self.bias.requires_grad,
        )

    @staticmethod
    def _metadata(state):
        schema = dict(
            format_version=int,
            format=str,
            in_features=int,
            out_features=int,
            compute_dtype=str,
            bias=bool,
            bias_trainable=bool,
        )
        if (
            type(state) is not dict
            or set(state) != set(schema)
            or any(type(state[k]) is not t for k, t in schema.items())
        ):
            raise ValueError("Invalid frozen-W8 metadata schema")
        if (
            state["format_version"] != 1
            or state["format"] != "signed_int8_per_output_scale"
        ):
            raise ValueError("Unsupported frozen-W8 format version")
        quant._weight_shape((state["out_features"], state["in_features"]))
        if state["compute_dtype"] not in DTYPE_NAMES or (
            not state["bias"] and state["bias_trainable"]
        ):
            raise ValueError("Invalid frozen-W8 dtype or bias metadata")
        return state

    def set_extra_state(self, state):
        if self._metadata(state) != self.get_extra_state():
            raise ValueError("Frozen-W8 metadata does not match the destination module")

    def _payload(self, state, prefix):
        self.set_extra_state(state.get(prefix + "_extra_state"))
        required = {prefix + "qweight", prefix + "scales", prefix + "_extra_state"}
        if self.bias is not None:
            required.add(prefix + "bias")
        if {k for k in state if k.startswith(prefix)} != required:
            raise ValueError(
                "Frozen-W8 payload must contain exactly the versioned fields"
            )
        values, scales = state[prefix + "qweight"], state[prefix + "scales"]
        if (
            not isinstance(values, torch.Tensor)
            or values.layout != torch.strided
            or values.device.type not in ("cpu", "cuda")
            or values.dtype != torch.int8
            or values.shape != (self.out_features, self.in_features)
        ):
            raise ValueError("Invalid INT8 weight payload")
        if (
            not isinstance(scales, torch.Tensor)
            or scales.layout != torch.strided
            or scales.device != values.device
            or scales.dtype != torch.float32
            or scales.shape != (self.out_features,)
            or scales.requires_grad
        ):
            raise ValueError("Invalid FP32 scale payload")
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
                raise ValueError("Frozen-W8 loading requires assign=False")
            self._payload(state_dict, prefix)
        except (ValueError, TypeError, KeyError) as error:
            error_msgs.append(f"{prefix}frozen-W8 state refused: {error}")
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
            raise ValueError("Expected a frozen-W8 state mapping")
        meta = cls._metadata(state_dict.get("_extra_state"))
        result = cls(
            meta["in_features"],
            meta["out_features"],
            bias=meta["bias"],
            device=device,
            dtype=DTYPE_NAMES[meta["compute_dtype"]],
            bias_trainable=meta["bias_trainable"],
        )
        result.load_state_dict(state_dict)
        return result

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, dtype={self.compute_dtype}, bias={self.bias is not None}"
