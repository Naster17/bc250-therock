# SPDX-License-Identifier: MIT
"""Experimental small complex GPU Schur prototype; never installed as eig dispatch.

Matrix arithmetic stays on the current GPU stream. Python controls iterations and
reads scalar convergence tests. Bounded separated-spectrum right-vector recovery
is available; no CPU eigensolver, balancing, batching, gradients or performance
contract. See N23.md, N24.md and N27.md.
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path

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


def _qr_similarity_reference(t, q, shift, m):
    eye = torch.eye(m, dtype=t.dtype, device=t.device)
    rotation, _ = torch.linalg.qr(t[:m, :m]-shift*eye)
    updated = t.clone()
    updated[:m, :] = rotation.mH@t[:m, :]
    updated[:, :m] = updated[:, :m]@rotation
    q[:, :m] = q[:, :m]@rotation
    return updated, q


_QR_LIBRARY = None


def _qr_similarity_device(t, q, shift, m):
    """Private normalized, finite, order<=16 driver inputs; current stream only."""
    global _QR_LIBRARY
    if _QR_LIBRARY is None:
        library = ctypes.CDLL(str(Path(__file__).with_name('schur_qr_n27.so')))
        library.bc250_qr_step.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*3 + [ctypes.c_void_p]
        library.bc250_qr_step.restype = ctypes.c_int
        _QR_LIBRARY = library
    t, q, shift = t.contiguous(), q.contiguous(), shift.contiguous()
    updated, newq = torch.empty_like(t), torch.empty_like(q)
    status = _QR_LIBRARY.bc250_qr_step(t.data_ptr(), q.data_ptr(), shift.data_ptr(),
        updated.data_ptr(), newq.data_ptr(), t.shape[0], m,
        int(t.dtype == torch.complex128), torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f'N2.7 QR launch returned HIP status {status}')
    return updated, newq


_CONTROL_LIBRARY = None


def _control_step_device(t, q, state, max_iterations):
    """Private matching normalized tensors/state, already on the current stream."""
    global _CONTROL_LIBRARY
    if _CONTROL_LIBRARY is None:
        library = ctypes.CDLL(str(Path(__file__).with_name('schur_control_n28.so')))
        library.bc250_schur_control.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*3 + [ctypes.c_void_p]
        library.bc250_schur_control.restype = ctypes.c_int
        _CONTROL_LIBRARY = library
    t, q = t.contiguous(), q.contiguous()
    updated, newq = torch.empty_like(t), torch.empty_like(q)
    status = _CONTROL_LIBRARY.bc250_schur_control(t.data_ptr(), q.data_ptr(),
        updated.data_ptr(), newq.data_ptr(), state.data_ptr(), t.shape[0],
        max_iterations, int(t.dtype == torch.complex128),
        torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f'N2.8 control launch returned HIP status {status}')
    return updated, newq


def _schur_control_device(t, q, max_iterations):
    """Bounded device iteration with one five-int status transfer per launch."""
    state = torch.tensor([t.shape[0], 0, 0, 0, 0], dtype=torch.int32, device=t.device)
    for _ in range(max_iterations+1):
        t, q = _control_step_device(t, q, state, max_iterations)
        active, iterations, deflations, since_deflation, phase = state.cpu().tolist()
        if phase == 3:
            raise RuntimeError('Nonfinite iterate or unexpected Hessenberg fill')
        if phase in (1, 2):
            return t, q, 0 if phase == 1 else active, iterations, deflations
    raise RuntimeError('Device control exceeded its bounded launch count')


_CHUNK_LIBRARY = None


def _chunk_step_device(t, q, state, max_iterations, chunk_size):
    """Private normalized tensors/valid state; at most32 reused control steps."""
    global _CHUNK_LIBRARY
    if _CHUNK_LIBRARY is None:
        library = ctypes.CDLL(str(Path(__file__).with_name('schur_chunk_n29.so')))
        library.bc250_schur_chunk.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*4 + [ctypes.c_void_p]
        library.bc250_schur_chunk.restype = ctypes.c_int
        _CHUNK_LIBRARY = library
    t, q = t.contiguous(), q.contiguous()
    updated, newq = torch.empty_like(t), torch.empty_like(q)
    status = _CHUNK_LIBRARY.bc250_schur_chunk(t.data_ptr(), q.data_ptr(),
        updated.data_ptr(), newq.data_ptr(), state.data_ptr(), t.shape[0],
        max_iterations, chunk_size, int(t.dtype == torch.complex128),
        torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f'N2.9 chunk launch returned HIP status {status}')
    return updated, newq


def _schur_chunk_device(t, q, max_iterations, chunk_size):
    state = torch.tensor([t.shape[0], 0, 0, 0, 0], dtype=torch.int32, device=t.device)
    for _ in range((max_iterations+1+chunk_size-1)//chunk_size):
        t, q = _chunk_step_device(t, q, state, max_iterations, chunk_size)
        active, iterations, deflations, since_deflation, phase = state.cpu().tolist()
        if phase == 3:
            raise RuntimeError('Nonfinite iterate or unexpected Hessenberg fill')
        if phase in (1, 2):
            return t, q, 0 if phase == 1 else active, iterations, deflations
    raise RuntimeError('Device chunk exceeded its bounded launch count')


@torch.no_grad()
def schur(a, *, max_iterations=256, qr_backend="reference", chunk_size=16):
    """Bounded explicit shifted QR experiment; inspect info before using T's diagonal.

    Positive info is not success and triggers no hidden host fallback. Q contains
    Schur vectors, not right eigenvectors of the original nonsymmetric matrix.
    """
    # Device QR is an explicit experiment; the reference remains the default.
    _validate(a)
    if qr_backend not in ("reference", "device", "device_control", "device_chunked"):
        raise ValueError("QR backend must be reference, device, device_control or device_chunked")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 32:
        raise ValueError('Chunk size must be an integer within1..32')
    if qr_backend != "device_chunked" and chunk_size != 16:
        raise ValueError('Nondefault chunk size requires device_chunked backend')
    step = _qr_similarity_device if qr_backend == "device" else _qr_similarity_reference
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
        if qr_backend in ("device_control", "device_chunked"):
            if qr_backend == "device_control":
                t, q, info, iterations, deflations = _schur_control_device(t, q, max_iterations)
            else:
                t, q, info, iterations, deflations = _schur_chunk_device(t, q, max_iterations, chunk_size)
            original_units = torch.complex(t.real*scale, t.imag*scale)
            if not bool(torch.isfinite(original_units).all()):
                raise RuntimeError('Schur rescaling is not representable in the input dtype')
            return SchurResult(original_units, q, info, iterations, deflations)
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
            updated, q = step(t, q, shift, m)
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
        # Solve all eigenvector columns together, one triangular row at a time.
        # The checked global pairwise gap guarantees every active pivot exceeds
        # floor. Inactive/diagonal pivots are replaced by1 before division.
        y = torch.eye(n, dtype=t.dtype, device=t.device)
        columns = torch.arange(n, device=t.device)
        for i in range(n-2, -1, -1):
            active_columns = columns > i
            pivots = diagonal[i]-diagonal
            safe_pivots = torch.where(active_columns, pivots, torch.ones_like(pivots))
            rhs = scaled[i, i+1:]@y[i+1:, :]
            y[i, :] = torch.where(active_columns, -rhs/safe_pivots, y[i, :])
            # Homogeneous per-column rescaling preserves already solved rows.
            # With scaled |T|<=1, |y|<=1 and the gap bound, active row values
            # cannot overflow for this order<=16/dtype contract. Defer finite
            # checks to the end instead of synchronizing the host per element.
            magnitude = y.abs().amax(dim=0)
            y = y/magnitude[None, :]
        norms = y.norm(dim=0)
        if not bool(torch.isfinite(y).all()) or not bool((norms > finfo.tiny).all()):
            raise RuntimeError('Unstable triangular eigenvector recovery')
        y = y/norms[None, :]
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
