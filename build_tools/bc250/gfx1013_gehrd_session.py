# SPDX-License-Identifier: MIT
"""Explicit single-thread/device/stream GEHRD resources for the cap16 prototype."""

import ctypes
import os
import threading

import torch


def _status(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f"{operation} returned rocBLAS status {code}")


class GehrdBindings:
    """Own CDLL references and typed function pointers, without a GPU handle."""

    def __init__(self) -> None:
        self.blas = ctypes.CDLL("librocblas.so")
        self.solver = ctypes.CDLL("librocsolver.so")
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.create = self.blas.rocblas_create_handle
        self.create.argtypes = [ctypes.POINTER(ptr)]
        self.create.restype = integer
        self.set_stream = self.blas.rocblas_set_stream
        self.set_stream.argtypes = [ptr, ptr]
        self.set_stream.restype = integer
        self.destroy = self.blas.rocblas_destroy_handle
        self.destroy.argtypes = [ptr]
        self.destroy.restype = integer
        self.gehrd = {}
        for dtype, name in (
            (torch.complex64, "rocsolver_cgehrd"),
            (torch.complex128, "rocsolver_zgehrd"),
        ):
            function = getattr(self.solver, name)
            function.argtypes = [ptr, integer, integer, integer, ptr, integer, ptr]
            function.restype = integer
            self.gehrd[dtype] = function


class GehrdSession:
    """Caller-owned handle; use a context manager or explicitly close it.

    No global handle cache, stream switching, shared-thread use or fork reuse.
    close() synchronizes the pinned stream before destroying native resources.
    """

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._thread = threading.get_ident()
        self.device = torch.device("cuda", torch.cuda.current_device())
        if (
            torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
            != "gfx1013"
        ):
            raise ValueError("GEHRD session qualified target is gfx1013 only")
        self.stream = torch.cuda.current_stream(self.device)
        self.bindings = GehrdBindings()
        self.handle = ctypes.c_void_p()
        self.closed = True
        self._busy = False
        self._entered = False
        self.uses = 0
        _status(self.bindings.create(ctypes.byref(self.handle)), "create_handle")
        try:
            _status(
                self.bindings.set_stream(
                    self.handle, ctypes.c_void_p(self.stream.cuda_stream)
                ),
                "set_stream",
            )
        except BaseException:
            _status(self.bindings.destroy(self.handle), "destroy_handle")
            self.handle = ctypes.c_void_p()
            raise
        self.closed = False

    def _check_owner(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("GEHRD session cannot be reused after fork")
        if threading.get_ident() != self._thread:
            raise RuntimeError("GEHRD session is pinned to its creating host thread")

    def _check_context(self, device: torch.device) -> None:
        self._check_owner()  # Check PID/thread before any GPU call.
        if self.closed:
            raise RuntimeError("GEHRD session is closed")
        if device != self.device:
            raise ValueError("GEHRD session device mismatch")
        if (
            torch.cuda.current_stream(self.device).cuda_stream
            != self.stream.cuda_stream
        ):
            raise RuntimeError("GEHRD session is pinned to its creating GPU stream")
        if self._busy:
            raise RuntimeError("GEHRD session is already in use")

    def _acquire(self, device: torch.device) -> None:
        self._check_context(device)
        self._busy = True

    def _release(self) -> None:
        self._busy = False

    def close(self) -> None:
        self._check_owner()
        if self.closed:
            return
        if self._busy:
            raise RuntimeError("Cannot close an active GEHRD session")
        with torch.cuda.device(self.device):
            self.stream.synchronize()
            _status(self.bindings.destroy(self.handle), "destroy_handle")
        self.handle = ctypes.c_void_p()
        self.closed = True

    def __enter__(self):
        self._check_context(self.device)
        if self._entered:
            raise RuntimeError("GEHRD session context cannot be nested")
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
