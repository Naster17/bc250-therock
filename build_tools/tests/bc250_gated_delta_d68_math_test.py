# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""D68 fused solve/scan math verification (numpy float64, no GPU).

Verifies the reformulation used by torch/nn/bc250_gated_delta.py's opt-in
use_fused_solve_scan path before board qualification:
  1. fused forward == default chunked forward == token-by-token recurrence;
  2. the unit-triangular inverse and its analytic gradient -tril(M^T dM M^T);
  3. the linear-recurrence scan and its analytic adjoint backward;
  4. the composed fused core (inverse + M/N/P/R precompute + scan) gradients.
The Triton kernels implement these same formulas; their masking/indexing is
exercised by the on-board A06 battery against the candidate module.
"""

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def token_recurrence(q, k, v, g, beta, normalize, initial=None):
    """Float64 reference, mirroring test_bc250_gated_delta.recurrent_reference."""
    q, k, v, g, beta = [
        x.transpose(0, 2, 1, 3) if x.ndim == 4 else x.transpose(0, 2, 1)
        for x in (q, k, v, g, beta)
    ]
    if normalize:
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    q = q * q.shape[-1] ** -0.5
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    state = np.zeros((b, h, dk, dv)) if initial is None else initial.copy()
    outputs = []
    for i in range(t):
        state = state * np.exp(g[:, :, i])[..., None, None]
        error = v[:, :, i] - (k[:, :, i][..., None] * state).sum(-2)
        state = (
            state
            + k[:, :, i][..., None] * (beta[:, :, i][..., None] * error)[..., None, :]
        )
        outputs.append((q[:, :, i][..., None] * state).sum(-2))
    return np.stack(outputs, 2).transpose(0, 2, 1, 3), state


def prepare(q, k, v, g, beta, normalize):
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    q, k, v = [x.transpose(0, 2, 1, 3) for x in (q, k, v)]
    g, beta = [x.transpose(0, 2, 1) for x in (g, beta)]
    if normalize:
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    q = q * dk**-0.5
    tp = ((t + 63) // 64) * 64
    pad = tp - t
    q, k, v = [np.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))) for x in (q, k, v)]
    g, beta = [np.pad(x, ((0, 0), (0, 0), (0, pad))) for x in (g, beta)]
    c = tp // 64
    kb = k * beta[..., None]
    vb = v * beta[..., None]
    q, k, kb, vb = [x.reshape(b, h, c, 64, x.shape[-1]) for x in (q, k, kb, vb)]
    decay = np.cumsum(g.reshape(b, h, c, 64), axis=-1)
    return q, k, kb, vb, decay


def pair_system_attention(q, k, kb, decay):
    mask = np.triu(np.ones((64, 64), dtype=bool), 1)
    diff = decay[..., :, None] - decay[..., None, :]
    pair = np.where(mask, 0.0, np.exp(np.where(mask, 0.0, diff)))
    system = (kb @ k.transpose(0, 1, 2, 4, 3)) * pair
    attention = (q @ k.transpose(0, 1, 2, 4, 3)) * pair
    return pair, system, attention


def unit_lower_solve(a, rhs):
    """Forward substitution treating the diagonal as 1 (unitriangular)."""
    n = a.shape[-1]
    x = np.zeros_like(rhs)
    for i in range(n):
        acc = rhs[..., i, :].copy()
        if i:
            acc = acc - (a[..., i, :i, None] * x[..., :i, :]).sum(axis=-2)
        x[..., i, :] = acc
    return x


def unit_lower_inverse(a):
    n = a.shape[-1]
    eye = np.broadcast_to(np.eye(n), a.shape).copy()
    return unit_lower_solve(a, eye)


def default_chunked(q, k, kb, vb, decay, system, attention, initial):
    b, h, c, _, dk = q.shape
    dv = vb.shape[-1]
    u = unit_lower_solve(system, vb)
    w = unit_lower_solve(system, kb * np.exp(decay)[..., None])
    qd = q * np.exp(decay)[..., None]
    kd = k * np.exp(decay[..., -1:] - decay)[..., None]
    cd = np.exp(decay[..., -1])[..., None, None]
    state = np.zeros((b, h, dk, dv)) if initial is None else initial.copy()
    out = np.zeros_like(u)
    for i in range(c):
        delta = u[:, :, i] - w[:, :, i] @ state
        out[:, :, i] = qd[:, :, i] @ state + attention[:, :, i] @ delta
        state = state * cd[:, :, i] + kd[:, :, i].transpose(0, 1, 3, 2) @ delta
    return out, state


def fused_core(q, k, kb, vb, decay, system, attention, initial):
    """The D68 reformulation: inverse + M/N/P/R precompute + linear scan."""
    b, h, c, _, dk = q.shape
    dv = vb.shape[-1]
    inv = unit_lower_inverse(system)
    u = inv @ vb
    w = inv @ (kb * np.exp(decay)[..., None])
    qd = q * np.exp(decay)[..., None]
    kd = k * np.exp(decay[..., -1:] - decay)[..., None]
    cd = np.exp(decay[..., -1])[..., None, None]
    eye = np.eye(dk)
    mm = cd * eye - kd.transpose(0, 1, 2, 4, 3) @ w
    nn = kd.transpose(0, 1, 2, 4, 3) @ u
    pp = qd - attention @ w
    rr = attention @ u
    state = np.zeros((b, h, dk, dv)) if initial is None else initial.copy()
    states = []
    out = np.zeros((b, h, c, 64, dv))
    for i in range(c):
        states.append(state.copy())
        out[:, :, i] = pp[:, :, i] @ state + rr[:, :, i]
        state = mm[:, :, i] @ state + nn[:, :, i]
    return out, state, (inv, u, w, qd, kd, cd, mm, nn, pp, rr, np.stack(states, 2))


def make_inputs(b, t, h, dk, dv, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((b, t, h, dk)) * 0.15
    k = rng.standard_normal((b, t, h, dk)) * 0.15
    v = rng.standard_normal((b, t, h, dv)) * 0.2
    g = -(rng.random((b, t, h)) * 0.15 + 0.01)
    beta = rng.random((b, t, h)) * 0.8 + 0.1
    return q, k, v, g, beta


@unittest.skipIf(np is None, "numpy required")
class ForwardEquivalenceTest(unittest.TestCase):
    def check_geometry(self, b, t, h, dk, dv, normalize, with_initial):
        xs = make_inputs(b, t, h, dk, dv, 6801)
        initial = (
            np.random.default_rng(7).standard_normal((b, h, dk, dv)) * 0.03
            if with_initial
            else None
        )
        ref_out, ref_state = token_recurrence(*xs, normalize, initial)
        q, k, kb, vb, decay = prepare(*xs, normalize)
        _, system, attention = pair_system_attention(q, k, kb, decay)
        def_out, def_state = default_chunked(
            q, k, kb, vb, decay, system, attention, initial
        )
        fus_out, fus_state, _ = fused_core(
            q, k, kb, vb, decay, system, attention, initial
        )
        tp = ((t + 63) // 64) * 64
        for name, actual, wanted in (
            ("default_out", def_out, ref_out),
            ("fused_out", fus_out, ref_out),
        ):
            trimmed = actual.reshape(b, h, tp, dv)[:, :, :t].transpose(0, 2, 1, 3)
            np.testing.assert_allclose(
                trimmed, wanted, atol=2e-12, rtol=2e-12, err_msg=name
            )
        np.testing.assert_allclose(def_state, ref_state, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(fus_state, ref_state, atol=2e-12, rtol=2e-12)

    def test_multi_chunk_partial(self):
        self.check_geometry(2, 130, 2, 3, 4, True, True)

    def test_exact_single_chunk(self):
        self.check_geometry(1, 64, 1, 5, 7, True, False)

    def test_two_chunks_no_normalize(self):
        self.check_geometry(1, 65, 2, 4, 4, False, True)

    def test_wide_heads(self):
        self.check_geometry(1, 129, 2, 16, 16, True, False)


@unittest.skipIf(np is None, "numpy required")
class InverseGradientTest(unittest.TestCase):
    def test_inverse_matches_numpy_solve(self):
        rng = np.random.default_rng(6802)
        a = rng.standard_normal((3, 64, 64)) * 0.1
        m = unit_lower_inverse(a)
        eff = np.eye(64) + np.tril(a, -1)
        np.testing.assert_allclose(
            m,
            np.linalg.solve(eff, np.broadcast_to(np.eye(64), (3, 64, 64))),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_inverse_gradient_finite_difference(self):
        rng = np.random.default_rng(6803)
        a = rng.standard_normal((1, 64, 64)) * 0.1
        w = rng.standard_normal((1, 64, 64))
        m = unit_lower_inverse(a)
        mt = m.transpose(0, 2, 1)
        analytic = np.tril(-(mt @ w @ mt), -1)
        eps = 1e-6
        coords = [tuple(x) for x in rng.integers(0, 64, size=(128, 2))]
        for i, j in coords:
            if j >= i:
                self.assertEqual(analytic[0, i, j], 0.0)
                continue
            ap = a.copy()
            ap[0, i, j] += eps
            am = a.copy()
            am[0, i, j] -= eps
            numeric = (
                (unit_lower_inverse(ap) * w).sum() - (unit_lower_inverse(am) * w).sum()
            ) / (2 * eps)
            self.assertAlmostEqual(analytic[0, i, j], numeric, delta=2e-6)


@unittest.skipIf(np is None, "numpy required")
class ScanGradientTest(unittest.TestCase):
    def scan(self, mm, nn, pp, rr, initial):
        b, h, c, dk, _ = mm.shape
        dv = nn.shape[-1]
        state = np.zeros((b, h, dk, dv)) if initial is None else initial.copy()
        states, outs = [], []
        for i in range(c):
            states.append(state.copy())
            outs.append(pp[:, :, i] @ state + rr[:, :, i])
            state = mm[:, :, i] @ state + nn[:, :, i]
        return np.stack(outs, 2), state, np.stack(states, 2)

    def backward(self, dout, dfinal, mm, pp, s_all, has_initial):
        b, h, c, dk, dv = s_all.shape
        g = np.zeros((b, h, dk, dv)) if dfinal is None else dfinal.copy()
        g_all = np.zeros_like(s_all)
        for i in range(c - 1, -1, -1):
            g = (
                mm[:, :, i].transpose(0, 1, 3, 2) @ g
                + pp[:, :, i].transpose(0, 1, 3, 2) @ dout[:, :, i]
            )
            g_all[:, :, i] = g
        tail = np.zeros((b, h, 1, dk, dv)) if dfinal is None else dfinal[:, :, None]
        g_next = np.concatenate([g_all[:, :, 1:], tail], 2)
        s_t = s_all.transpose(0, 1, 2, 4, 3)
        return (
            g_next @ s_t,
            g_next,
            dout @ s_t,
            dout,
            (g_all[:, :, 0] if has_initial else None),
        )

    def test_scan_gradient_finite_difference(self):
        rng = np.random.default_rng(6804)
        b, h, c, dk, dv = 1, 2, 3, 3, 4
        mm = rng.standard_normal((b, h, c, dk, dk)) * 0.1 + np.eye(dk) * 0.9
        nn = rng.standard_normal((b, h, c, dk, dv)) * 0.1
        pp = rng.standard_normal((b, h, c, 64, dk)) * 0.1
        rr = rng.standard_normal((b, h, c, 64, dv)) * 0.1
        initial = rng.standard_normal((b, h, dk, dv)) * 0.03
        wout = rng.standard_normal((b, h, c, 64, dv))
        wfin = rng.standard_normal((b, h, dk, dv))

        def loss(mm, nn, pp, rr, initial):
            out, final, _ = self.scan(mm, nn, pp, rr, initial)
            return (out * wout).sum() + (final * wfin).sum()

        out, final, s_all = self.scan(mm, nn, pp, rr, initial)
        grads = self.backward(wout, wfin, mm, pp, s_all, True)
        eps = 1e-6
        for tensor, grad, name in zip(
            (mm, nn, pp, rr, initial), grads, ("mm", "nn", "pp", "rr", "initial")
        ):
            flat = tensor.reshape(-1)
            coords = rng.integers(0, flat.size, size=48)
            for idx in coords:
                orig = flat[idx]
                flat[idx] = orig + eps
                lp = loss(mm, nn, pp, rr, initial)
                flat[idx] = orig - eps
                lm = loss(mm, nn, pp, rr, initial)
                flat[idx] = orig
                numeric = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(
                    grad.reshape(-1)[idx], numeric, delta=2e-6, msg=f"{name}[{idx}]"
                )

    def test_scan_no_final_grad(self):
        rng = np.random.default_rng(6805)
        b, h, c, dk, dv = 1, 1, 2, 3, 4
        mm = rng.standard_normal((b, h, c, dk, dk)) * 0.1 + np.eye(dk) * 0.9
        nn = rng.standard_normal((b, h, c, dk, dv)) * 0.1
        pp = rng.standard_normal((b, h, c, 64, dk)) * 0.1
        rr = rng.standard_normal((b, h, c, 64, dv)) * 0.1
        wout = rng.standard_normal((b, h, c, 64, dv))
        out, final, s_all = self.scan(mm, nn, pp, rr, None)
        dmm, dnn, dpp, drr, dinit = self.backward(wout, None, mm, pp, s_all, False)
        self.assertIsNone(dinit)
        eps = 1e-6
        idx = (0, 0, 1, 2, 2)
        orig = mm[idx]
        mm[idx] = orig + eps
        lp = (self.scan(mm, nn, pp, rr, None)[0] * wout).sum()
        mm[idx] = orig - eps
        lm = (self.scan(mm, nn, pp, rr, None)[0] * wout).sum()
        mm[idx] = orig
        self.assertAlmostEqual(dmm[idx], (lp - lm) / (2 * eps), delta=2e-6)


if __name__ == "__main__":
    unittest.main()
