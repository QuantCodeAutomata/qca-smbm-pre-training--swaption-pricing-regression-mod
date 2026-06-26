"""Tests for src/bootstrap.py — coterminal swap rate bootstrap."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bootstrap import bootstrap_single, bootstrap_batch, log_abar_jacobian


# ---------------------------------------------------------------------------
# Single-path bootstrap
# ---------------------------------------------------------------------------

class TestBootstrapSingle:
    """Validate analytical properties of the bootstrap formula."""

    def _default_S(self, n: int = 5) -> np.ndarray:
        return np.full(n, 0.02)

    def test_returns_correct_shapes(self):
        S = self._default_S(5)
        P_bar, A_bar = bootstrap_single(S, delta=1.0)
        assert P_bar.shape == (6,), "P_bar must have n_active+1 elements"
        assert A_bar.shape == (5,), "A_bar must have n_active elements"

    def test_terminal_discount_factor_equals_one(self):
        """P^e / P^e = 1.0 always (terminal node)."""
        S = self._default_S(5)
        P_bar, _ = bootstrap_single(S, delta=1.0)
        assert P_bar[-1] == pytest.approx(1.0, abs=1e-14)

    def test_last_annuity_equals_delta(self):
        """A_bar^{e-1,e} = delta (single period annuity)."""
        delta = 1.0
        S = self._default_S(5)
        _, A_bar = bootstrap_single(S, delta=delta)
        assert A_bar[-1] == pytest.approx(delta, abs=1e-14)

    def test_positivity_positive_rates(self):
        """All discount and annuity factors must be positive for S > 0."""
        S = self._default_S(5)
        P_bar, A_bar = bootstrap_single(S, delta=1.0)
        assert np.all(P_bar > 0), "All P_bar must be positive"
        assert np.all(A_bar > 0), "All A_bar must be positive"

    def test_decreasing_discount_factors(self):
        """Relative discount factors P^i/P^e should be non-decreasing in i (farther maturity costs less)."""
        S = self._default_S(5)
        P_bar, _ = bootstrap_single(S, delta=1.0)
        # P_bar[0] >= P_bar[1] >= ... >= P_bar[4] >= P_bar[5]=1 is wrong
        # Actually P_bar[k] = P^{first+k} / P^e, so P_bar[0] > P_bar[1] > ... > 1
        # because closer maturity discounts less
        assert P_bar[0] >= P_bar[-1], "P_bar monotonically decreasing toward terminal"
        for k in range(len(P_bar) - 1):
            assert P_bar[k] >= P_bar[k + 1] - 1e-12

    def test_swap_rate_consistency(self):
        """Bootstrap is consistent: S^{0,e} = (P^0/P^e - 1) / A_bar^{0,e}."""
        S = np.array([0.015, 0.020, 0.025, 0.018, 0.022])
        delta = 1.0
        P_bar, A_bar = bootstrap_single(S, delta=delta)
        for k in range(5):
            S_implied = (P_bar[k] - P_bar[k + 1]) / (delta * P_bar[k + 1])
            # Rearranged: S^k = (P_bar[k] - 1) / A_bar[k] --- but that's the full swap
            # Actually: S^k * A_bar[k] = P_bar[k] - P_bar[-1]
            lhs = S[k] * A_bar[k]
            rhs = P_bar[k] - P_bar[-1]
            assert lhs == pytest.approx(rhs, rel=1e-8), f"Swap k={k}: consistency failed"

    def test_single_component(self):
        """With a single swap rate, A_bar = delta and P_bar = [1+S*delta, 1]."""
        S = np.array([0.02])
        delta = 1.0
        P_bar, A_bar = bootstrap_single(S, delta=delta)
        assert A_bar[0] == pytest.approx(delta, abs=1e-14)
        assert P_bar[0] == pytest.approx(1.0 + S[0] * delta, abs=1e-14)
        assert P_bar[1] == pytest.approx(1.0, abs=1e-14)

    def test_zero_swap_rate(self):
        """Zero swap rate → all P_bar = 1 (flat discount)."""
        S = np.zeros(5)
        P_bar, A_bar = bootstrap_single(S, delta=1.0)
        assert np.all(P_bar == pytest.approx(1.0, abs=1e-14))

    def test_smbm_initial_rates(self):
        """Bootstrap at SMBM initial rates must return positive factors."""
        from src.params import S0
        S = np.full(5, S0)
        P_bar, A_bar = bootstrap_single(S, delta=1.0)
        assert np.all(P_bar > 0) and np.all(A_bar > 0)


# ---------------------------------------------------------------------------
# Batch bootstrap
# ---------------------------------------------------------------------------

class TestBootstrapBatch:
    def test_batch_matches_single(self):
        """Vectorised batch must agree with single-path loop."""
        rng = np.random.default_rng(0)
        S_batch = rng.uniform(0.01, 0.04, size=(32, 5))
        P_batch, A_batch = bootstrap_batch(S_batch, delta=1.0)

        for n in range(32):
            P_s, A_s = bootstrap_single(S_batch[n], delta=1.0)
            assert P_batch[n] == pytest.approx(P_s, rel=1e-12)
            assert A_batch[n] == pytest.approx(A_s, rel=1e-12)

    def test_batch_shapes(self):
        S_batch = np.ones((100, 4)) * 0.02
        P_batch, A_batch = bootstrap_batch(S_batch, delta=1.0)
        assert P_batch.shape == (100, 5), "P_batch shape"
        assert A_batch.shape == (100, 4), "A_batch shape"

    def test_batch_positivity(self):
        rng = np.random.default_rng(7)
        S_batch = rng.uniform(0.005, 0.05, size=(64, 5))
        P_batch, A_batch = bootstrap_batch(S_batch, delta=1.0)
        assert np.all(P_batch > 0)
        assert np.all(A_batch > 0)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

class TestLogAbarJacobian:
    def test_shape(self):
        S = np.full(5, 0.02)
        jac = log_abar_jacobian(S, i=0, delta=1.0)
        assert jac.shape == (5,)

    def test_zeros_for_flat_zero_rates(self):
        """With zero rates, A_bar stays at delta; derivative w.r.t. any S is finite."""
        S = np.zeros(5)
        # All A_bar > 0, should not raise
        jac = log_abar_jacobian(S, i=0, delta=1.0)
        assert jac.shape == (5,)

    def test_finite_values(self):
        S = np.full(5, 0.02)
        jac = log_abar_jacobian(S, i=2, delta=1.0)
        assert np.all(np.isfinite(jac)), "Jacobian must be finite"

    def test_perturbation_monotonicity(self):
        """d(ln A_bar^0)/dS^j >= 0 for all j (increasing S increases annuity)."""
        S = np.full(5, 0.02)
        jac = log_abar_jacobian(S, i=0, delta=1.0)
        assert np.all(jac >= -1e-10), "Jacobian entries should be non-negative"
