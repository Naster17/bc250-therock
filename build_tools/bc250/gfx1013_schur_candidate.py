# SPDX-License-Identifier: MIT
"""Experimental small complex GPU Schur prototype; never installed as eig dispatch.

Matrix arithmetic stays on the current GPU stream. Python controls iterations and
reads scalar convergence tests. No CPU eigensolver, balancing, eigenvector
recovery, batching, gradients or performance contract. See N23.md.
"""
import ctypes
from dataclasses import dataclass

import torch


@dataclass
class SchurResult:
    t: torch.Tensor
    q: torch.Tensor
    info: int  # 0: converged; >0: leading active block still unconverged
    iterations: int
    deflations: int


def _validate(a):
    if not a.is_cuda or a.dtype not in (torch.complex64, torch.complex128):
        raise ValueError('Candidate requires a GPU complex64/complex128 matrix')
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not 1 <= a.shape[0] <= 16:
        raise ValueError('Candidate accepts one square matrix of order1..16')
    if torch.cuda.get_device_properties(a.device).gcnArchName.split(':')[0] != 'gfx1013':
        raise ValueError('Candidate qualified target is gfx1013 only')
    if a.requires_grad:
        raise ValueError('Candidate has no autograd contract')
    if not bool(torch.isfinite(a).all()):
        raise ValueError('Candidate requires finite input')


def _status(code, operation):
    if code != 0:
        raise RuntimeError(f'{operation} returned rocBLAS status {code}')


@torch.no_grad()
def hessenberg(a):
    """Return H,Q with A=Q H Q^H using native rocSOLVER GEHRD reflectors."""
    _validate(a)
    n = a.shape[0]
    with torch.cuda.device(a.device):
        if n == 1:
            return a.clone(), torch.eye(n, dtype=a.dtype, device=a.device)
        blas = ctypes.CDLL('librocblas.so')
        solver = ctypes.CDLL('librocsolver.so')
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        blas.rocblas_create_handle.argtypes = [ctypes.POINTER(ptr)]
        blas.rocblas_create_handle.restype = integer
        blas.rocblas_set_stream.argtypes = [ptr, ptr]
        blas.rocblas_set_stream.restype = integer
        blas.rocblas_destroy_handle.argtypes = [ptr]
        blas.rocblas_destroy_handle.restype = integer
        gehrd = getattr(solver, 'rocsolver_cgehrd' if a.dtype == torch.complex64 else 'rocsolver_zgehrd')
        gehrd.argtypes = [ptr, integer, integer, integer, ptr, integer, ptr]
        gehrd.restype = integer
        # Always own the overwritten matrix, even if the input is already column-major.
        packed = a.mT.clone(memory_format=torch.contiguous_format).mT
        tau = torch.empty(n-1, dtype=a.dtype, device=a.device)
        handle = ptr()
        _status(blas.rocblas_create_handle(ctypes.byref(handle)), 'create_handle')
        try:
            _status(blas.rocblas_set_stream(handle, ptr(torch.cuda.current_stream().cuda_stream)), 'set_stream')
            _status(gehrd(handle, n, 1, n, ptr(packed.data_ptr()), n, ptr(tau.data_ptr())), 'gehrd')
            q = torch.eye(n, dtype=a.dtype, device=a.device)
            for i in range(n-1):
                v = torch.cat((torch.zeros(i+1, dtype=a.dtype, device=a.device),
                               torch.ones(1, dtype=a.dtype, device=a.device), packed[i+2:, i]))
                q = q - tau[i]*(q@v)[:, None]*v.conj()[None, :]
            return torch.triu(packed, diagonal=-1), q
        finally:
            _status(blas.rocblas_destroy_handle(handle), 'destroy_handle')


@torch.no_grad()
def schur(a, *, max_iterations=256):
    """Bounded explicit shifted QR experiment; inspect info before using T's diagonal.

    Positive info is not success and triggers no hidden host fallback. Q contains
    Schur vectors, not right eigenvectors of the original nonsymmetric matrix.
    """
    _validate(a)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 0 <= max_iterations <= 4096:
        raise ValueError('Iteration budget must be an integer within0..4096')
    with torch.cuda.device(a.device):
        t, q = hessenberg(a)
        active = a.shape[0]
        iterations, deflations, since_deflation = 0, 0, 0
        finfo = torch.finfo(a.real.dtype)
        while active > 1:
            m = active
            threshold = finfo.eps*(t[m-2, m-2].abs()+t[m-1, m-1].abs())+finfo.tiny
            if bool(t[m-1, m-2].abs() <= threshold):
                t[m-1, m-2] = 0
                active -= 1
                deflations += 1
                since_deflation = 0
                continue
            if iterations == max_iterations:
                return SchurResult(t, q, active, iterations, deflations)
            aa, bb, cc, dd = t[m-2,m-2], t[m-2,m-1], t[m-1,m-2], t[m-1,m-1]
            middle = (aa+dd)/2
            root = torch.sqrt(((aa-dd)/2).square()+bb*cc)
            shift1, shift2 = middle+root, middle-root
            shift = shift1 if bool((shift1-dd).abs() <= (shift2-dd).abs()) else shift2
            # A deterministic exceptional shift bounds stagnation attempts; it is
            # not a convergence guarantee for arbitrary spectra.
            if since_deflation and since_deflation % 20 == 0:
                shift = dd+cc.abs()
            eye = torch.eye(m, dtype=a.dtype, device=a.device)
            rotation, _ = torch.linalg.qr(t[:m, :m]-shift*eye)
            updated = t.clone()
            updated[:m, :] = rotation.mH@t[:m, :]
            updated[:, :m] = updated[:, :m]@rotation
            q[:, :m] = q[:, :m]@rotation
            # Exact QR preserves Hessenberg structure; remove only roundoff fill.
            fill = torch.tril(updated, diagonal=-2).norm()
            if not bool(torch.isfinite(updated).all()) or not bool(fill <= 128*finfo.eps*updated.norm()):
                raise RuntimeError('Nonfinite iterate or unexpected Hessenberg fill')
            t = torch.triu(updated, diagonal=-1)
            iterations += 1
            since_deflation += 1
        return SchurResult(t, q, 0, iterations, deflations)
