"""Tests for src/swaption_pricing.py — exp_1 regression model validation."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import XI0, T_EXERCISE, STRIKES, OMEGA, S0, DELTA_T, RHO_SW
from src.swaption_pricing import (
    sigma_t,
    simulate_smbm_annuity,
    compute_swaption_price,
    compute_swaption_grid,
    fit_swaption_model,
    price_swaption,
)
from src.black_scholes import bs_receiver_swaption


# ---------------------------------------------------------------------------
# sigma_t function
# ---------------------------------------------------------------------------

class TestSigmaT:
    """Paper formula: sigma_t = sqrt(xi0) * exp((omega/2)*X - (omega^2/4)*t)."""

    def test_at_origin(self):
        """At X=0, t=0: sigma = sqrt(xi0)."""
        xi0, omega = XI0[0], OMEGA
        result = sigma_t(xi0_i=xi0, X_t=np.array([0.0]), t=0.0, omega=omega)
        assert result[0] == pytest.approx(np.sqrt(xi0), rel=1e-10)

    def test_positive(self):
        """sigma must be strictly positive for any (X, t)."""
        for i in range(5):
            for X_val in [-2.0, 0.0, 2.0]:
                for t_val in [0.0, 2.0, T_EXERCISE[i]]:
                    result = sigma_t(xi0_i=XI0[i], X_t=np.array([X_val]), t=t_val, omega=OMEGA)
                    assert result[0] > 0, f"sigma_t must be positive (i={i}, X={X_val}, t={t_val})"

    def test_decreasing_in_t_for_x_zero(self):
        """For X=0: sigma_t = sqrt(xi0)*exp(-omega^2*t/4) decreases in t."""
        xi0, omega = XI0[0], OMEGA
        s0 = sigma_t(xi0_i=xi0, X_t=np.array([0.0]), t=0.0, omega=omega)[0]
        s1 = sigma_t(xi0_i=xi0, X_t=np.array([0.0]), t=10.0, omega=omega)[0]
        assert s1 < s0, "sigma_t should decrease over time at X=0"

    def test_formula_numerics(self):
        """Explicit formula check: sigma = sqrt(xi0) * exp((omega/2)*X - (omega^2/4)*t)."""
        xi0, omega = 1.287e-4, 0.8403
        X, t = 1.5, 3.0
        expected = np.sqrt(xi0) * np.exp((omega / 2) * X - (omega ** 2 / 4) * t)
        result = sigma_t(xi0_i=xi0, X_t=np.array([X]), t=t, omega=omega)[0]
        assert result == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# simulate_smbm_annuity
# ---------------------------------------------------------------------------

class TestSimulateSMBMAnnuity:
    """Inner MC under annuity measure A^{i,e}: returns terminal S values."""

    def test_output_shape(self):
        """Returned array must have length N_inner."""
        N = 200
        S_paths = simulate_smbm_annuity(
            S0_grid=0.02, X0_grid=0.0, tau=4.0,
            xi0_i=XI0[0], omega=OMEGA, rho=RHO_SW,
            N_inner=N, seed=5,
        )
        assert len(S_paths) == N

    def test_terminal_swap_martingale(self):
        """Under annuity measure, S is a martingale; check mean ≈ S0."""
        xi0 = XI0[0]
        tau = 4.0
        S_paths = simulate_smbm_annuity(
            S0_grid=0.02, X0_grid=0.0, tau=tau,
            xi0_i=xi0, omega=OMEGA, rho=RHO_SW,
            N_inner=5000, seed=1,
        )
        assert np.mean(S_paths) == pytest.approx(0.02, abs=0.005), "S must be martingale"

    def test_payoffs_derived_non_negative(self):
        """Receiver payoff max(k-S, 0) must always be >= 0."""
        S_paths = simulate_smbm_annuity(
            S0_grid=0.02, X0_grid=0.0, tau=4.0,
            xi0_i=XI0[0], omega=OMEGA, rho=RHO_SW,
            N_inner=500, seed=42,
        )
        payoffs = np.maximum(0.02 - S_paths, 0.0)
        assert np.all(payoffs >= 0), "Receiver payoffs must be non-negative"

    def test_returns_ndarray(self):
        """simulate_smbm_annuity must return numpy array."""
        S = simulate_smbm_annuity(0.02, 0.0, 4.0, XI0[0], N_inner=100, seed=0)
        assert isinstance(S, np.ndarray)


# ---------------------------------------------------------------------------
# compute_swaption_price
# ---------------------------------------------------------------------------

class TestComputeSwaptionPrice:
    """Wrapper that returns the mean payoff = v_est."""

    def test_positive_for_atm(self):
        price = compute_swaption_price(
            S0_grid=0.02, X0_grid=0.0, tau=4.0, strike=0.02,
            xi0_i=XI0[0], N_inner=500, seed=42,
        )
        assert price > 0

    def test_non_negative(self):
        for strike in [0.01, 0.015, 0.02]:
            price = compute_swaption_price(
                S0_grid=0.02, X0_grid=0.0, tau=4.0, strike=strike,
                xi0_i=XI0[0], N_inner=300, seed=1,
            )
            assert price >= 0.0

    def test_monotone_in_strike(self):
        """Higher strike → higher receiver price."""
        prices = [
            compute_swaption_price(0.02, 0.0, 4.0, k, XI0[0], N_inner=800, seed=3)
            for k in [0.010, 0.015, 0.020]
        ]
        assert prices[0] <= prices[1] <= prices[2] + 1e-6

    def test_bs_consistency_coarse(self):
        """ATM price should be within 20% of Black-Scholes value at X0=0, kappa=0."""
        tau = T_EXERCISE[0]
        xi0 = XI0[0]
        bs_val = bs_receiver_swaption(S=0.02, K=0.02, integrated_var=xi0 * tau)
        mc_val = compute_swaption_price(
            S0_grid=0.02, X0_grid=0.0, tau=tau, strike=0.02,
            xi0_i=xi0, N_inner=5000, seed=999,
        )
        rel_err = abs(mc_val - bs_val) / bs_val
        assert rel_err < 0.20, f"MC vs BS relative error {rel_err:.2%} > 20%"

    def test_zero_tau_returns_intrinsic(self):
        """At tau=0 (< 1e-6) price must equal max(k-S, 0)."""
        price_itm = compute_swaption_price(0.015, 0.0, 0.0, 0.02, XI0[0], N_inner=10)
        assert price_itm == pytest.approx(0.02 - 0.015, rel=1e-9)
        price_otm = compute_swaption_price(0.025, 0.0, 0.0, 0.02, XI0[0], N_inner=10)
        assert price_otm == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# compute_swaption_grid
# ---------------------------------------------------------------------------

class TestComputeSwaptionGrid:
    """Grid generation for regression training."""

    def test_output_shapes(self):
        i, k = 0, 0.02
        S_n, X_n, tau_n, V = compute_swaption_grid(
            component_idx=i, strike=k, N_grid=4, N_inner=100
        )
        assert S_n.shape == (4,)
        assert X_n.shape == (4,)
        assert tau_n.shape == (4,)
        assert V.shape == (4, 4, 4)

    def test_grid_non_negative(self):
        _, _, _, V = compute_swaption_grid(
            component_idx=0, strike=0.02, N_grid=4, N_inner=100
        )
        assert np.all(V >= 0), "Swaption prices on grid must be non-negative"

    def test_tau_grid_range(self):
        """tau grid must span [Delta_t, T_i]."""
        i = 0
        S_n, X_n, tau_n, V = compute_swaption_grid(
            component_idx=i, strike=0.02, N_grid=5, N_inner=100
        )
        T_i = T_EXERCISE[i]
        assert tau_n[0] == pytest.approx(DELTA_T, rel=1e-6)
        assert tau_n[-1] == pytest.approx(T_i, rel=1e-6)

    def test_s_grid_range(self):
        """S grid must be centred around S0 with width 5*sigma_i."""
        i = 0
        xi0_i = XI0[i]
        T_i = T_EXERCISE[i]
        sigma_i = np.sqrt(xi0_i * T_i)
        S_nodes, _, _, _ = compute_swaption_grid(
            component_idx=i, strike=0.02, N_grid=5, N_inner=100
        )
        expected_min = max(0.001, 0.02 - 5 * sigma_i)
        expected_max = 0.02 + 5 * sigma_i
        assert S_nodes[0] == pytest.approx(expected_min, rel=1e-6)
        assert S_nodes[-1] == pytest.approx(expected_max, rel=1e-6)


# ---------------------------------------------------------------------------
# fit_swaption_model / price_swaption
# ---------------------------------------------------------------------------

class TestFitAndPriceModel:
    """Regression model accuracy tests.
    
    price_swaption(model_data, S_t, X_t, tau) → scalar float (single point).
    """

    def test_model_atm_positive(self, tiny_swaption_model):
        """ATM interpolated price at midpoint tau must be positive."""
        md = tiny_swaption_model
        tau_mid = 0.5 * (md['tau_nodes'][0] + md['tau_nodes'][-1])
        price = price_swaption(md, S_t=0.02, X_t=0.0, tau=tau_mid)
        assert price > 0, "ATM model price must be positive"

    def test_model_returns_non_negative(self, tiny_swaption_model):
        """Interpolated prices must be >= 0 (clipped internally)."""
        md = tiny_swaption_model
        S_q = np.linspace(md['S_nodes'][0], md['S_nodes'][-1], 3)
        X_q = np.linspace(md['X_nodes'][0], md['X_nodes'][-1], 3)
        tau_q = np.linspace(md['tau_nodes'][0], md['tau_nodes'][-1], 3)
        for s in S_q:
            for x in X_q:
                for t in tau_q:
                    p = price_swaption(md, S_t=float(s), X_t=float(x), tau=float(t))
                    assert p >= 0.0, f"price_swaption must return >= 0 (got {p} at S={s}, X={x}, tau={t})"

    def test_boundary_intrinsic_itm(self, tiny_swaption_model):
        """At tau <= 1e-6, must equal max(k-S, 0) exactly."""
        md = tiny_swaption_model
        strike = md['strike']
        s_itm = strike - 0.008  # deeply ITM
        p = price_swaption(md, S_t=s_itm, X_t=0.0, tau=0.0)
        expected = max(strike - s_itm, 0.0)
        assert p == pytest.approx(expected, rel=1e-9)

    def test_boundary_intrinsic_otm(self, tiny_swaption_model):
        """At tau=0, OTM swaption must be exactly 0."""
        md = tiny_swaption_model
        strike = md['strike']
        p = price_swaption(md, S_t=strike + 0.005, X_t=0.0, tau=0.0)
        assert p == pytest.approx(0.0, abs=1e-12)

    def test_model_returns_finite_scalar(self, tiny_swaption_model):
        """price_swaption must return a finite scalar."""
        md = tiny_swaption_model
        for S_val in [0.015, 0.02, 0.025]:
            p = price_swaption(md, S_t=S_val, X_t=0.0, tau=md['tau_nodes'][-1])
            assert np.isfinite(p), f"Model price must be finite at S={S_val}"


# ---------------------------------------------------------------------------
# Methodology adherence: paper constants
# ---------------------------------------------------------------------------

class TestPaperConstants:
    """Verify paper parameters are correctly encoded."""

    def test_omega(self):
        assert OMEGA == pytest.approx(0.8403, rel=1e-6)

    def test_xi0_values(self):
        expected = [1.132e-04, 1.228e-04, 1.287e-04, 1.344e-04, 1.451e-04]
        for i, val in enumerate(expected):
            assert XI0[i] == pytest.approx(val, rel=1e-4)

    def test_t_exercise(self):
        expected = [4, 5, 6, 7, 8]
        for i, val in enumerate(expected):
            assert T_EXERCISE[i] == val

    def test_rho_sw(self):
        assert RHO_SW == pytest.approx(-0.259, rel=1e-4)

    def test_s0(self):
        assert S0 == pytest.approx(0.02, rel=1e-9)

    def test_delta_t(self):
        assert DELTA_T == pytest.approx(1.0 / 32, rel=1e-9)

    def test_strikes(self):
        assert set(STRIKES) == {0.01, 0.015, 0.02}
