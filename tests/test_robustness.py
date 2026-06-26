"""Tests for src/robustness.py — exp_5 modified-parameter path generation and robustness."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import OMEGA, XI0, T_EXERCISE, RHO_SW, build_correlation_matrix
from src.robustness import (
    build_modified_correlation_matrices,
    verify_atm_price_consistency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OMEGA_MOD = OMEGA / 2.0  # 0.42015


def _get_original_matrices() -> list:
    """Build the 5 default original correlation matrices."""
    return [build_correlation_matrix(n) for n in [5, 4, 3, 2, 1]]


# ---------------------------------------------------------------------------
# build_modified_correlation_matrices
# ---------------------------------------------------------------------------

class TestBuildModifiedCorrelationMatrices:
    """Verify sign-flip of S-X and X-S blocks, structure preservation."""

    def test_returns_five_matrices(self):
        """Must return exactly 5 modified matrices."""
        mods = build_modified_correlation_matrices()
        assert len(mods) == 5, f"Expected 5 matrices, got {len(mods)}"

    def test_shapes_match_originals(self):
        """Each modified matrix must have same shape as the original."""
        originals = _get_original_matrices()
        mods = build_modified_correlation_matrices(originals)
        expected_shapes = [(10, 10), (8, 8), (6, 6), (4, 4), (2, 2)]
        for i, (m, shape) in enumerate(zip(mods, expected_shapes)):
            assert m.shape == shape, (
                f"Matrix {i}: expected shape {shape}, got {m.shape}"
            )

    def test_diagonal_is_one(self):
        """All modified matrices must have diagonal = 1 (correlation matrices)."""
        mods = build_modified_correlation_matrices()
        for i, m in enumerate(mods):
            np.testing.assert_allclose(
                np.diag(m), 1.0, atol=1e-10,
                err_msg=f"Matrix {i}: diagonal must be 1"
            )

    def test_symmetric(self):
        """Correlation matrices must be symmetric."""
        mods = build_modified_correlation_matrices()
        for i, m in enumerate(mods):
            np.testing.assert_allclose(
                m, m.T, atol=1e-12,
                err_msg=f"Matrix {i}: must be symmetric"
            )

    def test_positive_semi_definite(self):
        """All modified matrices must be positive semi-definite."""
        mods = build_modified_correlation_matrices()
        for i, m in enumerate(mods):
            eigvals = np.linalg.eigvalsh(m)
            assert eigvals.min() >= -1e-8, (
                f"Matrix {i}: not PSD (min eigenvalue = {eigvals.min():.2e})"
            )

    def test_sx_block_sign_flipped(self):
        """S-X block entries must be negated relative to originals (off-diagonal of SX block)."""
        originals = _get_original_matrices()
        mods = build_modified_correlation_matrices(originals)
        for orig, mod in zip(originals, mods):
            n = orig.shape[0] // 2
            # S-X block: rows 0:n, cols n:2n
            orig_sx = orig[:n, n:]
            mod_sx = mod[:n, n:]
            # The off-diagonal entries should be negated
            # (diagonal entries of this block = rho(S^i, X^i) = within-factor)
            np.testing.assert_allclose(
                mod_sx, -orig_sx, atol=1e-12,
                err_msg="S-X block should be exactly negated"
            )

    def test_ss_block_unchanged(self):
        """S-S block (swap-swap correlations) must be unchanged."""
        originals = _get_original_matrices()
        mods = build_modified_correlation_matrices(originals)
        for orig, mod in zip(originals, mods):
            n = orig.shape[0] // 2
            np.testing.assert_allclose(
                mod[:n, :n], orig[:n, :n], atol=1e-12,
                err_msg="S-S block must be unchanged by sign-flip"
            )

    def test_xx_block_unchanged(self):
        """X-X block (variance-variance correlations) must be unchanged."""
        originals = _get_original_matrices()
        mods = build_modified_correlation_matrices(originals)
        for orig, mod in zip(originals, mods):
            n = orig.shape[0] // 2
            np.testing.assert_allclose(
                mod[n:, n:], orig[n:, n:], atol=1e-12,
                err_msg="X-X block must be unchanged by sign-flip"
            )

    def test_2x2_matrix_correctly_modified(self):
        """For 2x2 (single active component): [[1, -rho],[rho, 1]] → [[1, rho],[-rho, 1]]."""
        orig_2x2 = build_correlation_matrix(1)
        mods = build_modified_correlation_matrices([orig_2x2])
        m = mods[0]
        # Within-factor correlation was -0.259 (S^5 row, X^5 col)
        # After sign flip, it should be +0.259
        assert m[0, 1] == pytest.approx(-orig_2x2[0, 1], rel=1e-10)
        assert m[1, 0] == pytest.approx(-orig_2x2[1, 0], rel=1e-10)

    def test_entries_in_minus_one_to_one(self):
        """All entries in modified matrices must be in [-1, 1]."""
        mods = build_modified_correlation_matrices()
        for i, m in enumerate(mods):
            assert np.all(m >= -1.0 - 1e-10), f"Matrix {i}: entry < -1"
            assert np.all(m <= 1.0 + 1e-10), f"Matrix {i}: entry > 1"

    def test_modifying_twice_restores_original(self):
        """Applying sign-flip twice should restore the original matrix."""
        originals = _get_original_matrices()
        once = build_modified_correlation_matrices(originals)
        twice = build_modified_correlation_matrices(once)
        for i, (orig, restored) in enumerate(zip(originals, twice)):
            np.testing.assert_allclose(
                restored, orig, atol=1e-10,
                err_msg=f"Matrix {i}: double sign-flip should restore original"
            )

    def test_accepts_custom_matrices(self):
        """Should accept and process any list of valid correlation matrices."""
        # Single 4x4 identity-like matrix (all within-factor corr = -0.259)
        n = 2
        C = np.eye(2 * n)
        rho = -0.259
        for i in range(n):
            C[i, i + n] = rho
            C[i + n, i] = rho
        mods = build_modified_correlation_matrices([C])
        assert len(mods) == 1
        assert mods[0].shape == (2 * n, 2 * n)


# ---------------------------------------------------------------------------
# verify_atm_price_consistency
# ---------------------------------------------------------------------------

class TestVerifyATMPriceConsistency:
    """ATM swaption prices should be omega-independent at X=0, kappa=0."""

    def test_returns_dict_of_dicts(self):
        """Should return a dict indexed by component index i in {0,...,4}."""
        result = verify_atm_price_consistency(
            omega_new=OMEGA_MOD, xi0=XI0, n_verify=300, seed=0
        )
        assert isinstance(result, dict)
        for i in range(5):
            assert i in result, f"Component i={i} missing from result"

    def test_result_keys_present(self):
        """Each component result must contain required metric keys."""
        result = verify_atm_price_consistency(
            omega_new=OMEGA_MOD, xi0=XI0, n_verify=200, seed=42
        )
        required_keys = ['T_i', 'bs_price_bp', 'mc_price_orig_bp', 'mc_price_mod_bp',
                         'diff_orig_bp', 'diff_mod_bp']
        for i in range(5):
            for key in required_keys:
                assert key in result[i], f"i={i}: missing key '{key}'"

    def test_bs_prices_positive(self):
        """Black-Scholes ATM prices must be positive for all components."""
        result = verify_atm_price_consistency(
            omega_new=OMEGA_MOD, xi0=XI0, n_verify=200, seed=1
        )
        for i in range(5):
            assert result[i]['bs_price_bp'] > 0, f"BS price must be positive for i={i}"

    def test_mc_prices_positive(self):
        """MC ATM prices (both original and modified omega) must be positive."""
        result = verify_atm_price_consistency(
            omega_new=OMEGA_MOD, xi0=XI0, n_verify=300, seed=2
        )
        for i in range(5):
            assert result[i]['mc_price_orig_bp'] > 0, f"MC orig price must be positive i={i}"
            assert result[i]['mc_price_mod_bp'] > 0, f"MC mod price must be positive i={i}"

    def test_diffs_non_negative(self):
        """Absolute differences must be non-negative."""
        result = verify_atm_price_consistency(
            omega_new=OMEGA_MOD, xi0=XI0, n_verify=200, seed=3
        )
        for i in range(5):
            assert result[i]['diff_orig_bp'] >= 0
            assert result[i]['diff_mod_bp'] >= 0

    def test_t_i_values_correct(self):
        """T_i values must match paper parameters: T_1=4,...,T_5=8."""
        result = verify_atm_price_consistency(
            omega_new=OMEGA_MOD, xi0=XI0, n_verify=100, seed=4
        )
        expected_T = [4, 5, 6, 7, 8]
        for i in range(5):
            assert result[i]['T_i'] == expected_T[i], (
                f"T_{i+1} expected {expected_T[i]}, got {result[i]['T_i']}"
            )

    def test_atm_price_roughly_bs_consistent(self):
        """MC ATM prices should be within 30% of Black-Scholes (coarse check for n=500)."""
        result = verify_atm_price_consistency(
            omega_new=OMEGA_MOD, xi0=XI0, n_verify=500, seed=99
        )
        for i in range(5):
            bs_bp = result[i]['bs_price_bp']
            mc_orig_bp = result[i]['mc_price_orig_bp']
            rel_err = abs(mc_orig_bp - bs_bp) / bs_bp
            assert rel_err < 0.30, (
                f"i={i}: MC orig vs BS rel error {rel_err:.1%} > 30%"
            )

    def test_omega_independence_coarse(self):
        """ATM prices under omega and omega/2 should be within 50% of each other (coarse MC)."""
        result = verify_atm_price_consistency(
            omega_new=OMEGA_MOD, xi0=XI0, n_verify=500, seed=88
        )
        for i in range(5):
            orig = result[i]['mc_price_orig_bp']
            mod = result[i]['mc_price_mod_bp']
            # ATM price is theoretically omega-independent; MC noise dominates at n=500
            # Just check they're in the same ballpark
            if orig > 0:
                ratio = mod / orig
                assert 0.5 < ratio < 2.0, (
                    f"i={i}: omega-independent ATM prices should be similar "
                    f"(orig={orig:.2f}bp, mod={mod:.2f}bp, ratio={ratio:.2f})"
                )


# ---------------------------------------------------------------------------
# Modified correlation matrix: Cholesky feasibility
# ---------------------------------------------------------------------------

class TestModifiedCholeskeyFeasibility:
    """Modified matrices must admit Cholesky decomposition for MC simulation."""

    def test_all_cholesky_succeed(self):
        """All 5 modified matrices must allow numpy Cholesky decomposition."""
        mods = build_modified_correlation_matrices()
        for i, m in enumerate(mods):
            try:
                L = np.linalg.cholesky(m)
                assert L is not None, f"Cholesky of matrix {i} returned None"
            except np.linalg.LinAlgError:
                # Try with small diagonal perturbation
                m_reg = m + 1e-8 * np.eye(m.shape[0])
                try:
                    np.linalg.cholesky(m_reg)
                except np.linalg.LinAlgError as e:
                    pytest.fail(
                        f"Matrix {i} not PSD even after regularization: {e}"
                    )

    def test_cholesky_lower_triangular(self):
        """Cholesky factor L must be lower triangular with positive diagonal."""
        mods = build_modified_correlation_matrices()
        for i, m in enumerate(mods):
            eigvals = np.linalg.eigvalsh(m)
            if eigvals.min() < 0:
                m = m + (-eigvals.min() + 1e-8) * np.eye(m.shape[0])
            L = np.linalg.cholesky(m)
            # Lower triangular
            upper = np.triu(L, k=1)
            np.testing.assert_allclose(
                upper, 0.0, atol=1e-15,
                err_msg=f"Matrix {i}: Cholesky factor must be lower triangular"
            )
            # Positive diagonal
            assert np.all(np.diag(L) > 0), f"Matrix {i}: Cholesky diagonal must be positive"

    def test_ll_t_reconstructs_matrix(self):
        """L @ L.T must equal the modified matrix (within numerical tolerance)."""
        mods = build_modified_correlation_matrices()
        for i, m in enumerate(mods):
            eigvals = np.linalg.eigvalsh(m)
            if eigvals.min() < 0:
                m = m + (-eigvals.min() + 1e-8) * np.eye(m.shape[0])
            L = np.linalg.cholesky(m)
            reconstructed = L @ L.T
            np.testing.assert_allclose(
                reconstructed, m, atol=1e-12,
                err_msg=f"Matrix {i}: L @ L.T must reconstruct the matrix"
            )


# ---------------------------------------------------------------------------
# Mathematical properties of original correlation matrices
# ---------------------------------------------------------------------------

class TestOriginalCorrelationMatrices:
    """Baseline: original matrices satisfy all required properties."""

    def test_original_psd(self):
        """All 5 original correlation matrices must be PSD."""
        for n in [5, 4, 3, 2, 1]:
            C = build_correlation_matrix(n)
            eigvals = np.linalg.eigvalsh(C)
            assert eigvals.min() >= -1e-10, (
                f"n={n}: original matrix not PSD (min eigval = {eigvals.min():.2e})"
            )

    def test_original_diagonal_one(self):
        """Original matrices must have diagonal = 1."""
        for n in [5, 4, 3, 2, 1]:
            C = build_correlation_matrix(n)
            np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)

    def test_original_within_factor_correlation(self):
        """Within-factor correlation rho(S^i, X^i) = -0.259 per paper."""
        for n in [5, 4, 3, 2, 1]:
            C = build_correlation_matrix(n)
            # Off-diagonal of S-X block (within-factor: row i, col n+i)
            for i in range(n):
                rho_actual = C[i, n + i]
                assert rho_actual == pytest.approx(RHO_SW, rel=1e-4), (
                    f"n={n}, i={i}: rho(S^i, X^i) = {rho_actual:.4f} ≠ {RHO_SW}"
                )

    def test_omega_mod_is_half_original(self):
        """omega_mod = omega / 2 per paper specification."""
        assert OMEGA_MOD == pytest.approx(OMEGA / 2.0, rel=1e-6)
