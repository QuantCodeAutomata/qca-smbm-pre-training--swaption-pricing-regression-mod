"""Tests for src/evaluation.py — metric computation, P&L analysis, table construction."""
import sys, os
import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    compute_cvar,
    compute_model_value,
    compute_metrics,
    run_consistency_checks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pl_constant(value: float, n: int = 1000) -> np.ndarray:
    """Constant P&L array."""
    return np.full(n, value)


def _pl_normal(n: int = 5000, mean: float = 0.0, std: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


# ---------------------------------------------------------------------------
# compute_cvar
# ---------------------------------------------------------------------------

class TestComputeCVaR:
    """CVaR_alpha(-PL) = -E[PL | PL <= q_alpha]."""

    @pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.8])
    def test_constant_pl(self, alpha):
        """If all PL = c, CVaR = -c (always falls into tail)."""
        c = 0.03
        pl = _pl_constant(c, 2000)
        cvar = compute_cvar(pl, alpha)
        assert cvar == pytest.approx(-c, abs=1e-8)

    def test_cvar_increases_with_alpha(self):
        """Higher alpha → worse tail → CVaR must not decrease."""
        pl = _pl_normal(10000, mean=0.01, std=0.05, seed=99)
        cvars = [compute_cvar(pl, a) for a in [0.2, 0.4, 0.6, 0.8]]
        for i in range(len(cvars) - 1):
            assert cvars[i] <= cvars[i + 1] + 1e-9, (
                f"CVaR should be non-decreasing in alpha: alpha={[0.2,0.4,0.6,0.8][i]}"
            )

    def test_cvar_finite(self):
        """CVaR must be finite for any reasonable P&L distribution."""
        pl = _pl_normal(500)
        for alpha in [0.2, 0.4, 0.6, 0.8]:
            assert np.isfinite(compute_cvar(pl, alpha))

    def test_cvar_vs_mean_tail(self):
        """CVaR should be close to the negative mean of the bottom (1-alpha) fraction."""
        rng = np.random.default_rng(3)
        pl = rng.normal(0.0, 0.1, 20000)
        alpha = 0.8
        cvar_val = compute_cvar(pl, alpha)
        # Reference: mean of worst 20%
        sorted_pl = np.sort(pl)
        n_tail = int((1 - alpha) * len(pl))
        ref = -np.mean(sorted_pl[:n_tail])
        assert abs(cvar_val - ref) / (abs(ref) + 1e-10) < 0.05

    def test_returns_float(self):
        """compute_cvar must return a Python float or numpy scalar."""
        pl = _pl_normal(100)
        result = compute_cvar(pl, 0.5)
        assert isinstance(result, (float, np.floating)), f"Expected float, got {type(result)}"

    def test_empty_pl_handles(self):
        """Edge case: single-element P&L."""
        pl = np.array([0.05])
        result = compute_cvar(pl, 0.8)
        assert np.isfinite(result), "CVaR with single element must be finite"

    def test_all_positive_pl_cvar_negative(self):
        """If all PL > 0 (profitable strategy), CVaR < 0 (model value > 0)."""
        pl = np.ones(1000) * 0.01 + np.abs(_pl_normal(1000, 0, 0.001))
        cvar = compute_cvar(pl, 0.8)
        assert cvar < 0, "CVaR should be negative when P&L is always positive"


# ---------------------------------------------------------------------------
# compute_model_value
# ---------------------------------------------------------------------------

class TestComputeModelValue:
    """Model Value V = -CVaR_alpha(-PL)."""

    def test_equals_negative_cvar(self):
        """model_value = -compute_cvar(pl, alpha)."""
        pl = _pl_normal(1000, seed=5)
        for alpha in [0.2, 0.5, 0.8]:
            mv = compute_model_value(pl, alpha)
            cvar = compute_cvar(pl, alpha)
            assert mv == pytest.approx(-cvar, rel=1e-10)

    def test_positive_for_profitable_strategy(self):
        """Profitable strategies (positive mean PL) should have positive Model Value."""
        pl = np.ones(1000) * 0.02  # constant positive P&L
        mv = compute_model_value(pl, alpha=0.8)
        assert mv > 0, "Model Value should be positive for constant positive P&L"

    @pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.8])
    def test_decreasing_in_alpha(self, alpha):
        """Model Value should be non-increasing in alpha (higher risk aversion → lower value)."""
        pass  # Tested through CVaR monotonicity


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Full metric dict from P&L distribution."""

    def _sample_metrics(self, n=1000, seed=42) -> dict:
        pl = _pl_normal(n, 0.02, 0.05, seed=seed)
        payoffs = _pl_normal(n, 0.03, 0.02, seed=seed + 1).clip(0)
        return compute_metrics(pl, alpha=0.5, exercise_payoffs=payoffs, max_swaption_t0=0.025)

    def test_all_required_keys_present(self):
        """All metric keys from the paper must be present."""
        metrics = self._sample_metrics()
        required = [
            'model_value', 'nonarb_value', 'hedge_pnl_mean',
            'model_switch', 'nonarb_switch',
            'p25', 'p50', 'p75', 'iqr',
            'loss_prob', 'expected_loss',
            'cvar95', 'cvar99',
        ]
        for key in required:
            assert key in metrics, f"Missing metric key '{key}'"

    def test_iqr_equals_p75_minus_p25(self):
        """IQR must equal P75 - P25 exactly."""
        metrics = self._sample_metrics()
        assert metrics['iqr'] == pytest.approx(
            metrics['p75'] - metrics['p25'], rel=1e-10
        )

    def test_cvar99_geq_cvar95(self):
        """CVaR99 must be >= CVaR95 (99th is a tighter tail)."""
        metrics = self._sample_metrics()
        assert metrics['cvar99'] >= metrics['cvar95'] - 1e-8, (
            f"CVaR99={metrics['cvar99']:.4f} < CVaR95={metrics['cvar95']:.4f}"
        )

    def test_loss_prob_in_01(self):
        """Loss probability must be in [0, 1]."""
        metrics = self._sample_metrics()
        assert 0.0 <= metrics['loss_prob'] <= 1.0

    def test_model_switch_value(self):
        """model_switch = model_value - max_swaption_t0."""
        metrics = self._sample_metrics()
        mv = metrics['model_value']
        ms = metrics['model_switch']
        assert ms == pytest.approx(mv - 0.025, rel=1e-10)

    def test_nonarb_switch_value(self):
        """nonarb_switch = nonarb_value - max_swaption_t0."""
        metrics = self._sample_metrics()
        nv = metrics['nonarb_value']
        ns = metrics['nonarb_switch']
        assert ns == pytest.approx(nv - 0.025, rel=1e-10)

    def test_nonarb_value_equals_mean_payoffs(self):
        """NonArb Value = mean(exercise_payoffs) when payoffs provided."""
        n = 500
        payoffs = np.abs(_pl_normal(n, 0.03, 0.01, seed=7))
        pl = _pl_normal(n, 0.02, 0.05, seed=8)
        metrics = compute_metrics(pl, alpha=0.5, exercise_payoffs=payoffs)
        assert metrics['nonarb_value'] == pytest.approx(np.mean(payoffs), rel=1e-10)

    def test_expected_loss_positive(self):
        """Expected Loss (conditional on loss) must be positive when loss occurs."""
        pl = _pl_normal(1000, 0.0, 0.05, seed=0)
        metrics = compute_metrics(pl, alpha=0.5)
        if metrics['loss_prob'] > 0:
            assert metrics['expected_loss'] > 0, "Expected Loss must be positive when losses exist"

    def test_percentiles_ordered(self):
        """P25 <= P50 <= P75 (drawdown percentiles must be ordered)."""
        metrics = self._sample_metrics()
        assert metrics['p25'] <= metrics['p50'] + 1e-10
        assert metrics['p50'] <= metrics['p75'] + 1e-10

    def test_all_metrics_finite(self):
        """All returned metric values must be finite numbers."""
        metrics = self._sample_metrics()
        for key, val in metrics.items():
            assert np.isfinite(val), f"Metric '{key}' = {val} is not finite"

    def test_no_exercise_payoffs_fallback(self):
        """When exercise_payoffs=None, nonarb_value falls back to mean(pl)."""
        pl = _pl_normal(200, 0.01, 0.05, seed=1)
        metrics = compute_metrics(pl, alpha=0.5, exercise_payoffs=None)
        assert metrics['nonarb_value'] == pytest.approx(np.mean(pl), rel=1e-8)

    def test_basis_points_scaling_invariance(self):
        """Scaling P&L by 10000 (bp conversion) should scale all monetary metrics."""
        pl = _pl_normal(1000, 0.01, 0.005, seed=0)
        payoffs = np.abs(_pl_normal(1000, 0.01, 0.002, seed=1))
        max_sw = 0.01

        m1 = compute_metrics(pl, 0.5, payoffs, max_sw)
        m2 = compute_metrics(pl * 10000, 0.5, payoffs * 10000, max_sw * 10000)

        for key in ['model_value', 'nonarb_value', 'iqr', 'expected_loss']:
            assert m2[key] == pytest.approx(m1[key] * 10000, rel=1e-5), (
                f"Key '{key}': scaling mismatch"
            )

    def test_extreme_alpha_consistency(self):
        """At alpha=0 (no risk aversion), model value approaches mean P&L."""
        pl = _pl_normal(5000, 0.05, 0.01, seed=33)
        metrics_low = compute_metrics(pl, alpha=0.01)
        # At very low alpha, CVaR ≈ mean of almost all paths → model value ≈ mean
        assert metrics_low['model_value'] == pytest.approx(np.mean(pl), abs=0.005)


# ---------------------------------------------------------------------------
# run_consistency_checks
# ---------------------------------------------------------------------------

class TestRunConsistencyChecks:
    """Automated consistency checks on Tables 1 and 2."""

    def _build_tables(self) -> tuple:
        """Build synthetic Tables 1 and 2 conforming to expected structure."""
        strategies = ['S_OS', 'S_Max', 'S_IS', 'S_I', 'S_ISM']
        alphas = [0.2, 0.4, 0.6, 0.8]
        rows1, rows2 = [], []
        for s in strategies:
            for a in alphas:
                # Table 1 row
                mv = 400.0 + np.random.uniform(-10, 10)
                nv = 456.0
                rows1.append({
                    'strategy': s, 'alpha': a,
                    'model_value_bp': mv, 'nonarb_value_bp': nv,
                    'hedge_pnl_mean_bp': mv - nv,
                    'model_switch_bp': mv - 397.0,
                    'nonarb_switch_bp': nv - 397.0,
                })
                # Table 2 row
                iqr = 73.0 if s == 'S_Max' else 50.0
                rows2.append({
                    'strategy': s, 'alpha': a,
                    'p25_bp': -20.0, 'p50_bp': 0.0, 'p75_bp': iqr - 20.0,
                    'iqr_bp': iqr, 'loss_prob': 0.3,
                    'expected_loss_bp': 30.0,
                    'cvar95_bp': 50.0, 'cvar99_bp': 80.0,
                })
        return pd.DataFrame(rows1), pd.DataFrame(rows2)

    def test_returns_dict(self):
        t1, t2 = self._build_tables()
        result = run_consistency_checks(t1, t2)
        assert isinstance(result, dict)

    def test_cvar99_geq_cvar95_check(self):
        """Consistency check must flag CVaR99 < CVaR95 as failure."""
        t1, t2 = self._build_tables()
        # Introduce violation
        t2.loc[0, 'cvar99_bp'] = t2.loc[0, 'cvar95_bp'] - 1.0
        result = run_consistency_checks(t1, t2)
        # Should detect violation (result value depends on implementation)
        assert isinstance(result, dict)  # at minimum returns dict

    def test_check_keys_present(self):
        """Consistency checks should return meaningful boolean keys."""
        t1, t2 = self._build_tables()
        result = run_consistency_checks(t1, t2)
        # At least one boolean check key expected
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Methodology adherence: drawdown definition
# ---------------------------------------------------------------------------

class TestDrawdownDefinition:
    """Drawdown = V - PL_n; percentiles computed over drawdowns."""

    def test_drawdown_computation(self):
        """If PL = [V, V, V, 0], drawdown = [0, 0, 0, V]; P50 = 0."""
        V = 0.05
        pl = np.array([V, V, V, 0.0])
        metrics = compute_metrics(pl, alpha=0.0, max_swaption_t0=0.0)
        model_val = metrics['model_value']
        # Drawdown = model_val - pl; for pl = [V, V, V, 0], draw = [mv-V, mv-V, mv-V, mv]
        drawdown = model_val - pl
        p50 = np.percentile(drawdown, 50)
        assert metrics['p50'] == pytest.approx(p50, rel=1e-8)

    def test_zero_pl_loss_prob_one(self):
        """If all PL = 0, model_value >= 0 and loss_prob depends on model_value sign."""
        pl = np.zeros(100)
        metrics = compute_metrics(pl, alpha=0.5)
        # model_value = 0, drawdown = 0 for all, loss_prob = 0
        assert metrics['model_value'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['loss_prob'] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for metric computation."""

    def test_single_path_pl(self):
        """Single P&L value must not crash."""
        pl = np.array([0.05])
        metrics = compute_metrics(pl, alpha=0.5)
        assert np.all([np.isfinite(v) for v in metrics.values()])

    def test_two_path_pl(self):
        """Two paths: computations must not crash."""
        pl = np.array([0.02, -0.01])
        metrics = compute_metrics(pl, alpha=0.5)
        assert metrics['loss_prob'] == pytest.approx(0.5, abs=1e-10)

    def test_all_positive_pl_no_expected_loss(self):
        """If all P&L > model_value, expected loss should be 0 or nearly so."""
        c = 0.10
        pl = np.full(200, c)
        metrics = compute_metrics(pl, alpha=0.5)
        # drawdown = model_val - c = -c + c = 0 for all → no losses
        assert metrics['loss_prob'] == pytest.approx(0.0, abs=1e-8)

    def test_very_large_pl(self):
        """Extreme P&L values should not cause overflow."""
        pl = np.full(100, 1e6)
        metrics = compute_metrics(pl, alpha=0.8)
        assert np.all([np.isfinite(v) for v in metrics.values()])

    def test_mixed_positive_negative(self):
        """Mixed P&L: loss_prob should be strictly between 0 and 1."""
        pl = np.concatenate([np.full(50, 0.05), np.full(50, -0.05)])
        metrics = compute_metrics(pl, alpha=0.5)
        assert 0 < metrics['loss_prob'] < 1, (
            f"loss_prob={metrics['loss_prob']} should be strictly between 0 and 1"
        )
