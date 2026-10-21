# SPDX-License-Identifier: MIT
"""Explicit BF16 -> FP32 FFT adapters; no torch.fft monkeypatch or CPU fallback.

Use e.g. ``bc250_fft.rfft2(x)``. Only BF16 input tensors are cast, on their
existing device, via differentiable ``Tensor.to(float32)``. With no ``out``,
results use complex64 (complex transforms) or float32 (real inverse/Hermitian
transforms); leaf gradients retain the leaf dtype and its rounding. Other
inputs and arguments, including ``out``, follow the native torch.fft contracts.
This is FP32 compute, not native BF16 FFT, and adds a conversion allocation.
"""

from functools import wraps
import torch

__version__ = "0.1.0"
__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
]


def _promoted(operator):
    @wraps(operator)
    def transform(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.dtype == torch.bfloat16:
            input = input.to(dtype=torch.float32)
        return operator(input, *args, **kwargs)

    transform.__module__ = __name__
    transform.__doc__ = (
        "Explicit BF16-to-FP32 adapter; other inputs/arguments follow torch.fft.\n\n"
        + (operator.__doc__ or "")
    )
    return transform


fft = _promoted(torch.fft.fft)
ifft = _promoted(torch.fft.ifft)
fft2 = _promoted(torch.fft.fft2)
ifft2 = _promoted(torch.fft.ifft2)
fftn = _promoted(torch.fft.fftn)
ifftn = _promoted(torch.fft.ifftn)
rfft = _promoted(torch.fft.rfft)
irfft = _promoted(torch.fft.irfft)
rfft2 = _promoted(torch.fft.rfft2)
irfft2 = _promoted(torch.fft.irfft2)
rfftn = _promoted(torch.fft.rfftn)
irfftn = _promoted(torch.fft.irfftn)
hfft = _promoted(torch.fft.hfft)
ihfft = _promoted(torch.fft.ihfft)
hfft2 = _promoted(torch.fft.hfft2)
ihfft2 = _promoted(torch.fft.ihfft2)
hfftn = _promoted(torch.fft.hfftn)
ihfftn = _promoted(torch.fft.ihfftn)
