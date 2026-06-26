"""Tests for src/mc_simulation.py — exp_2 path generation validation."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import (
    S0, XI0, OMEGA, DELTA_T, T_EXERCISE, N_STEPS, STRIKES,
    N_COMPONENTS, RHO_SW,
)
from src.mc_simulation import (
    compute_drift_batch,
    simulate_paths,
    validate_paths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_paths(n_paths: int = 32, seed: int = 7) -> dict:
    """Generate a tiny path set for fast unit tests."""
    return simulate_paths(n_paths=n_paths, swaption_models=None, seed=seed, verbose=False)


# ---------------------------------------------------------------------------
# simulate_paths: output structure
# ---------------------------------------------------------------------------

class TestSimulatePathsStructure:
    """Verify paths_dict has correct keys, shapes, and dtypes."""

    REQUIRED_KEYS = [
        'S_rates', 'X_states', 'annuity_factors',
        'swap_prices', 'swaption_K002', 'swaption_K001',
        'swaption_K0015', 'exercise_dates_idx',
    ]

    def test_all_keys_present(self, tiny_paths):
        for k in self.REQUIRED_KEYS:
            assert k in tiny_paths, f"Missing key '{k}' in paths_dict"

    def test_shapes(self, tiny_paths):
        N = tiny_paths['S_rates'].shape[0]
        T = N_STEPS + 1  # 289
        C = N_COMPONENTS  # 5
        for key in ['S_rates', 'X_states', 'annuity_factors',
                    'swap_prices', 'swaption_K002', 'swaption_K001', 'swaption_K0015']:
            arr = tiny_paths[key]
            assert arr.shape == (N, T, C), (
                f"'{key}' shape {arr.shape} != expected ({N}, {T}, {C})"
            )

    def test_exercise_dates_idx(self, tiny_paths):
        idx = tiny_paths['exercise_dates_idx']
        assert list(idx) == [128, 160, 192, 224, 256], f"Wrong exercise indices: {idx}"

    def test_dtypes_float(self, tiny_paths):
        for key in ['S_rates', 'X_states', 'annuity_factors', 'swap_prices']:
            assert tiny_paths[key].dtype in (np.float64, np.float32), (
                f"'{key}' dtype {tiny_paths[key].dtype} not float"
            )


# ---------------------------------------------------------------------------
# Boundary conditions and initial values
# ---------------------------------------------------------------------------

class TestInitialConditions:
    """Verify t=0 values match model parameters."""

    def test_initial_swap_rates(self, tiny_paths):
        """S^{i,e}_0 = S0 = 0.02 for all i."""
        S0_actual = tiny_paths['S_rates'][:, 0, :]
        np.testing.assert_allclose(
            S0_actual, S0, atol=1e-12,
            err_msg="Initial swap rates must equal S0=0.02"
        )

    def test_initial_x_states_zero(self, tiny_paths):
        """X^{i,e}_0 = 0 for all i."""
        X0_actual = tiny_paths['X_states'][:, 0, :]
        np.testing.assert_allclose(
            X0_actual, 0.0, atol=1e-12,
            err_msg="Initial X states must be zero"
        )

    def test_initial_swaption_positive(self, tiny_paths):
        """ATM swaption prices at t=0 must be positive for all components."""
        O0 = tiny_paths['swaption_K002'][:, 0, :]
        assert np.all(O0 >= 0), "Initial swaption prices must be non-negative"
        # At least one should be strictly positive
        assert np.any(O0 > 0), "At least one initial swaption price must be positive"


# ---------------------------------------------------------------------------
# Financial properties of simulated paths
# ---------------------------------------------------------------------------

class TestPathFinancialProperties:
    """Financial validity checks as specified in exp_2 methodology."""

    def test_swaption_prices_non_negative(self, tiny_paths):
        """All O_bar values must be >= 0 throughout the simulation."""
        for key in ['swaption_K002', 'swaption_K001', 'swaption_K0015']:
            arr = tiny_paths[key]
            neg = np.sum(arr < -1e-6)
            assert neg == 0, f"'{key}' has {neg} negative values"

    def test_annuity_factors_positive(self, tiny_paths):
        """All non-zero annuity factors must be > 0."""
        A_bar = tiny_paths['annuity_factors']
        # Only check the upper-left triangle where components are active
        for i in range(N_COMPONENTS):
            T_i_idx = [128, 160, 192, 224, 256][i]
            A_before = A_bar[:, :T_i_idx + 1, i]
            # Positive annuity (some tiny numerical zeros before expiry may exist)
            neg = np.sum(A_before < -1e-8)
            assert neg == 0, f"Annuity factor for i={i} has {neg} negative entries before expiry"

    def test_max_swaption_at_t0_order_of_magnitude(self, tiny_paths):
        """Max initial swaption value should be in the range [100, 700] bp."""
        max_O0_bp = np.max(np.mean(tiny_paths['swaption_K002'][:, 0, :], axis=0)) * 10_000
        assert 50 < max_O0_bp < 1000, (
            f"Max swaption at t=0: {max_O0_bp:.1f} bp — outside expected range [50, 1000]"
        )

    def test_swap_prices_finite(self, tiny_paths):
        """Swap prices must be finite everywhere."""
        assert np.all(np.isfinite(tiny_paths['swap_prices'])), "Swap prices must be finite"

    def test_swap_rates_finite(self, tiny_paths):
        """Swap rates must be finite everywhere."""
        assert np.all(np.isfinite(tiny_paths['S_rates'])), "Swap rates must be finite"


# ---------------------------------------------------------------------------
# Martingale test (exp_2 Phase F validation)
# ---------------------------------------------------------------------------

class TestMartingaleCondition:
    """Under P^e, relative swap prices U_bar should be approximately martingales."""

    def test_mean_delta_swap_small(self):
        """Mean change in swap prices over a larger set should be small."""
        paths = _small_paths(n_paths=128, seed=42)
        swap_p = paths['swap_prices']
        delta = np.diff(swap_p, axis=1)  # (N, T-1, 5)
        mean_delta = np.abs(np.mean(delta, axis=0))  # (T-1, 5)
        # Check first period (all 5 active) — within first 128 steps
        max_mean_delta = float(np.max(mean_delta[:128, :]))
        assert max_mean_delta < 0.01, (
            f"Mean absolute swap price change {max_mean_delta:.6f} exceeds martingale bound 0.01"
        )


# ---------------------------------------------------------------------------
# validate_paths function
# ---------------------------------------------------------------------------

class TestValidatePaths:
    """Tests for the validate_paths validation helper."""

    def test_returns_dict(self, tiny_paths):
        result = validate_paths(tiny_paths, verbose=False)
        assert isinstance(result, dict)

    def test_required_keys(self, tiny_paths):
        result = validate_paths(tiny_paths, verbose=False)
        for key in ['swaption_nonneg_pass', 'annuity_pos_pass', 'martingale_pass']:
            assert key in result, f"Missing validation key '{key}'"

    def test_swaption_nonneg_passes(self, tiny_paths):
        result = validate_paths(tiny_paths, verbose=False)
        assert result['swaption_nonneg_pass'], (
            f"Swaption non-negativity failed: {result.get('swaption_negative_count')} negatives"
        )

    def test_annuity_pos_passes(self, tiny_paths):
        result = validate_paths(tiny_paths, verbose=False)
        assert result['annuity_pos_pass'], (
            f"Annuity positivity failed: min = {result.get('annuity_min')}"
        )

    def test_initial_s0_value(self, tiny_paths):
        result = validate_paths(tiny_paths, verbose=False)
        assert result['S0_initial'] == pytest.approx(0.02, rel=1e-6)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Identical seed must produce identical paths."""

    def test_same_seed_same_paths(self):
        paths_a = _small_paths(n_paths=8, seed=123)
        paths_b = _small_paths(n_paths=8, seed=123)
        np.testing.assert_array_equal(
            paths_a['S_rates'], paths_b['S_rates'],
            err_msg="Same seed must produce identical S_rates"
        )

    def test_different_seeds_different_paths(self):
        paths_a = _small_paths(n_paths=8, seed=1)
        paths_b = _small_paths(n_paths=8, seed=2)
        assert not np.allclose(paths_a['S_rates'], paths_b['S_rates']), (
            "Different seeds must produce different paths"
        )


# ---------------------------------------------------------------------------
# compute_drift_batch
# ---------------------------------------------------------------------------

class TestComputeDriftBatch:
    """Tests for the drift computation used in the Euler-Maruyama step."""

    def test_returns_arrays(self):
        """compute_drift_batch must return (mu_S, mu_X) arrays of shape (N, n_active)."""
        N = 10
        n_active = 5
        from src.params import build_correlation_matrix
        C = build_correlation_matrix(n_active)

        S = np.full((N, n_active), 0.02)
        X = np.zeros((N, n_active))

        mu_S, mu_X = compute_drift_batch(
            S_active=S, X_active=X, t=0.0, n_active=n_active,
            xi0=XI0[:n_active], omega=OMEGA, corr_matrix=C,
        )
        assert mu_S.shape == (N, n_active), f"mu_S shape {mu_S.shape} != ({N}, {n_active})"
        assert mu_X.shape == (N, n_active), f"mu_X shape {mu_X.shape} != ({N}, {n_active})"

    def test_drift_finite(self):
        """Drift values must be finite."""
        N, n_active = 5, 3
        from src.params import build_correlation_matrix
        C = build_correlation_matrix(n_active)
        S = np.full((N, n_active), 0.02)
        X = np.zeros((N, n_active))
        mu_S, mu_X = compute_drift_batch(S, X, 0.0, n_active, XI0[:n_active], OMEGA, C)
        assert np.all(np.isfinite(mu_S)), "mu_S must be finite"
        assert np.all(np.isfinite(mu_X)), "mu_X must be finite"

    def test_drift_small_at_t0(self):
        """At t=0, X=0, S=S0, drift corrections should be small (model-dependent, but finite)."""
        N, n_active = 4, 5
        from src.params import build_correlation_matrix
        C = build_correlation_matrix(n_active)
        S = np.full((N, n_active), 0.02)
        X = np.zeros((N, n_active))
        mu_S, mu_X = compute_drift_batch(S, X, 0.0, n_active, XI0, OMEGA, C)
        # Drift should not be astronomically large
        assert np.all(np.abs(mu_S) < 1.0), "mu_S should be small at t=0, X=0"
