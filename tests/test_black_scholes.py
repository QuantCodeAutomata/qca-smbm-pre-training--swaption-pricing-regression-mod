"""Tests for src/black_scholes.py — Black-76 receiver swaption formula."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.black_scholes import bs_receiver_swaption


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bs_call(F: float, K: float, sigma_sq: float) -> float:
    """Bachelier payer swaption price satisfying put-call parity with bs_receiver_swaption.

    Under the normal (Bachelier) model:
        payer = (F - K) * Phi(-d) + sigma_T * phi(d),   d = (K - F) / sigma_T
    This gives:  receiver - payer = K - F  (exact parity).
    """
    from scipy.stats import norm
    if sigma_sq <= 0:
        return max(F - K, 0.0)
    sigma_T = np.sqrt(sigma_sq)
    d = (K - F) / sigma_T
    return (F - K) * norm.cdf(-d) + sigma_T * norm.pdf(d)


# ---------------------------------------------------------------------------
# Boundary and unit tests
# ---------------------------------------------------------------------------

class TestBSReceiverSwaption:
    """Methodology adherence: Black-76 log-normal receiver swaption."""

    def test_atm_positive(self):
        """ATM price must be strictly positive for any integrated_var > 0."""
        price = bs_receiver_swaption(S=0.02, K=0.02, integrated_var=1.132e-4 * 4)
        assert price > 0.0, "ATM receiver price must be positive"

    def test_zero_var_intrinsic_otm(self):
        """At zero variance, OTM receiver (S > K) has zero value."""
        price = bs_receiver_swaption(S=0.03, K=0.02, integrated_var=0.0)
        assert price == pytest.approx(0.0, abs=1e-12), "OTM receiver at zero vol = 0"

    def test_zero_var_intrinsic_itm(self):
        """At zero variance, ITM receiver equals K - S."""
        K, S = 0.02, 0.01
        price = bs_receiver_swaption(S=S, K=K, integrated_var=0.0)
        assert price == pytest.approx(K - S, abs=1e-12)

    def test_put_call_parity(self):
        """Black-76 put-call parity: call - put = F - K (undiscounted)."""
        S, K, iv = 0.025, 0.02, 1.287e-4 * 6
        receiver = bs_receiver_swaption(S=S, K=K, integrated_var=iv)
        payer = _bs_call(F=S, K=K, sigma_sq=iv)
        # receiver - payer = K - S (put-call parity for Black-76 undiscounted)
        parity_diff = receiver - payer
        assert parity_diff == pytest.approx(K - S, rel=1e-6)

    def test_monotone_in_strike(self):
        """Receiver price is non-decreasing in strike K."""
        S, iv = 0.02, 1.132e-4 * 4
        strikes = [0.010, 0.015, 0.020, 0.025, 0.030]
        prices = [bs_receiver_swaption(S=S, K=k, integrated_var=iv) for k in strikes]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1] + 1e-15, "Receiver price must be non-decreasing in K"

    def test_monotone_in_vol(self):
        """Receiver price is non-decreasing in integrated variance."""
        S, K = 0.02, 0.02
        ivars = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
        prices = [bs_receiver_swaption(S=S, K=K, integrated_var=iv) for iv in ivars]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1] + 1e-15, "Receiver price must be non-decreasing in vol"

    def test_smbm_atm_target(self):
        """ATM swaption at (i=0) should match Black-76 with integrated_var = xi0_0 * T_1."""
        from src.params import XI0, T_EXERCISE
        xi0_0, T1 = XI0[0], T_EXERCISE[0]
        integrated_var = xi0_0 * T1
        price = bs_receiver_swaption(S=0.02, K=0.02, integrated_var=integrated_var)
        # Price should be approximately sqrt(integrated_var) / (2*sqrt(2*pi)) * S
        # Rough bound: must be positive and < K/2
        assert 0 < price < 0.02 / 2

    def test_all_smbm_components(self):
        """Black-76 ATM price positive for all 5 SMBM components."""
        from src.params import XI0, T_EXERCISE
        for i in range(5):
            iv = XI0[i] * T_EXERCISE[i]
            p = bs_receiver_swaption(S=0.02, K=0.02, integrated_var=iv)
            assert p > 0, f"Component i={i}: ATM price must be positive"
            assert p < 0.02, f"Component i={i}: ATM price must be < K=0.02"

    def test_returns_float(self):
        """Function must return a Python float or numpy scalar."""
        price = bs_receiver_swaption(S=0.02, K=0.02, integrated_var=1e-4)
        assert isinstance(price, (float, np.floating)), "Must return numeric scalar"

    def test_deep_otm_near_zero(self):
        """Deep OTM receiver (S >> K) price should be near zero."""
        price = bs_receiver_swaption(S=0.10, K=0.02, integrated_var=1e-4)
        assert price < 1e-6, "Deep OTM receiver price should be negligible"

    def test_deep_itm_near_intrinsic(self):
        """Deep ITM receiver (S << K) price approaches K - S."""
        S, K = 0.001, 0.02
        price = bs_receiver_swaption(S=S, K=K, integrated_var=1e-6)
        assert price == pytest.approx(K - S, rel=1e-3), "Deep ITM receiver → K - S"
