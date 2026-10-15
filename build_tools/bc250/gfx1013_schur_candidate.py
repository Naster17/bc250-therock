# SPDX-License-Identifier: MIT
"""Experimental small complex GPU Schur prototype; never installed as eig dispatch.

Matrix arithmetic stays on the current GPU stream. Python controls iterations and
reads scalar convergence tests. Bounded separated-spectrum right-vector recovery
is available; no CPU eigensolver, balancing, batching, gradients or performance
contract. See N23.md and N24.md.
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


def _normal_scale(a):
    """GPU component scaling avoids complex magnitude/division overflow.

    All-subnormal input is outside this prototype's contract. Detect nonzero
    subnormal bits even if hardware floating arithmetic flushes them to zero.
    """
    components = torch.view_as_real(a.resolve_conj().resolve_neg())
    scale = components.abs().amax()
    finfo = torch.finfo(a.real.dtype)
    if not bool(scale > 0):
        integer_dtype = torch.int32 if a.dtype == torch.complex64 else torch.int64
        bits = components.view(integer_dtype)
        if bool((bits & torch.iinfo(integer_dtype).max).any()):
            raise RuntimeError('Subnormal input scale unsupported by candidate')
        return a.clone(), scale
    if not bool(torch.isfinite(scale)) or not bool(scale >= finfo.tiny):
        raise RuntimeError('Subnormal or nonfinite input scale unsupported by candidate')
    unit = torch.complex(a.real/scale, a.imag/scale)
    if not bool(torch.isfinite(unit).all()):
        raise RuntimeError('Matrix normalization produced a nonfinite result')
    return unit, scale


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
        n = a.shape[0]
        if n == 1:
            return SchurResult(a.clone(), torch.eye(n, dtype=a.dtype, device=a.device), 0, 0, 0)
        unit, scale = _normal_scale(a)
        if not bool(scale > 0):
            return SchurResult(a.clone(), torch.eye(n, dtype=a.dtype, device=a.device), 0, 0, n-1)
        t, q = hessenberg(unit)
        active = n
        iterations, deflations, since_deflation = 0, 0, 0
        finfo = torch.finfo(a.real.dtype)

        def finish(info):
            original_units = torch.complex(t.real*scale, t.imag*scale)
            if not bool(torch.isfinite(original_units).all()):
                raise RuntimeError('Schur rescaling is not representable in the input dtype')
            return SchurResult(original_units, q, info, iterations, deflations)
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
                return finish(active)
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
        return finish(0)


@dataclass
class EigenvectorResult:
    values: torch.Tensor
    vectors: torch.Tensor  # unit-norm right eigenvectors, in matching Schur order
    relative_separation: float
    triangular_residual: float


@torch.no_grad()
def right_eigenvectors(result):
    """Recover separated-spectrum right vectors from a converged small Schur form.

    Refuse unconverged/nontriangular forms, repeated or close eigenvalues,
    nonunitary transforms and unstable recovery. No perturbing singular pivots,
    hidden host solver, gradients or guarantee for ill-conditioned eigenbases.
    """
    if not isinstance(result, SchurResult):
        raise ValueError('Expected a candidate SchurResult')
    if result.info != 0:
        raise RuntimeError('Cannot recover eigenvectors from an unconverged Schur result')
    t, q = result.t, result.q
    _validate(t)
    _validate(q)
    if q.shape != t.shape or q.dtype != t.dtype or q.device != t.device:
        raise ValueError('Schur T and Q must have matching shape, dtype and device')
    with torch.cuda.device(t.device):
        n = t.shape[0]
        finfo = torch.finfo(t.real.dtype)
        eye = torch.eye(n, dtype=t.dtype, device=t.device)
        if bool(torch.tril(t, diagonal=-1).abs().any()):
            raise ValueError('Eigenvector recovery requires exactly upper triangular T')
        if not bool((q.mH@q-eye).norm() <= 128*n*finfo.eps):
            raise ValueError('Schur Q is not sufficiently unitary')
        values = t.diag().clone()
        if n == 1:
            return EigenvectorResult(values, q/q.norm(), float('inf'), 0.)
        unit, scale = _normal_scale(t)
        if not bool(scale > 0):
            raise RuntimeError('Spectrum not sufficiently separated or scale unsupported')
        # Preserve the existing max-complex-magnitude separation policy, but only
        # compute complex magnitudes/divisions once values are safely near unity.
        scaled = unit/unit.abs().amax()
        diagonal = scaled.diag()
        gaps = (diagonal[:, None]-diagonal[None, :]).abs()
        gaps.fill_diagonal_(float('inf'))
        separation = gaps.min()
        floor = 8*finfo.eps**.5
        if not bool(separation > floor):
            raise RuntimeError('Spectrum not sufficiently separated for candidate recovery')
        y = torch.zeros_like(t)
        for j in range(n):
            column = torch.zeros(n, dtype=t.dtype, device=t.device)
            column[j] = 1
            for i in range(j-1, -1, -1):
                pivot = diagonal[i]-diagonal[j]
                if not bool(pivot.abs() > floor):
                    raise RuntimeError('Near-singular triangular recovery pivot')
                column[i] = -(scaled[i, i+1:j+1]@column[i+1:j+1])/pivot
                magnitude = column.abs().amax()
                if not bool(torch.isfinite(column).all()) or not bool(magnitude > 0):
                    raise RuntimeError('Unstable triangular eigenvector recovery')
                # Scaling a homogeneous solution preserves its direction and
                # bounds subsequent dot products; never perturb the pivot.
                column = column/magnitude
            y[:, j] = column/column.norm()
        residuals = (scaled@y-y*diagonal[None, :]).norm(dim=0)
        denominator = (scaled.norm()+diagonal.abs())*y.norm(dim=0)
        residual = (residuals/denominator).amax()
        if not bool(torch.isfinite(residual)) or not bool(residual <= 64*n*finfo.eps):
            raise RuntimeError('Triangular eigenvector residual exceeds candidate bound')
        vectors = q@y
        norms = vectors.norm(dim=0)
        if not bool(torch.isfinite(vectors).all()) or not bool((norms > finfo.tiny).all()):
            raise RuntimeError('Unstable Schur-vector transform')
        vectors = vectors/norms[None, :]
        return EigenvectorResult(values, vectors, float(separation), float(residual))
