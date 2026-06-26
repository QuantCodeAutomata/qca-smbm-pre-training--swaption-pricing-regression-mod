"""Bachelier (normal model) pricing for European receiver swaptions.

The SMBM dynamics are arithmetic: dS_t = sigma_t * dW_t (no drift under
the annuity measure A^{i,e}).  This means S_{T_i} | F_0 is approximately
normal, so the correct closed-form is the Bachelier formula, not
Black-Scholes (lognormal).

Bachelier formula:
    E[(K - S_T)^+] = (K - S) * Phi(d) + sigma_T * phi(d)
    where d = (K - S) / sigma_T,  sigma_T = sqrt(integrated_var).
"""
import numpy as np
from scipy.stats import norm


def bs_receiver_swaption(
    S: float,
    K: float,
    integrated_var: float,
) -> float:
    """Compute Bachelier price for a European receiver swaption.

    Under the annuity measure A^{i,e} with kappa=0 and X_0=0 the swap rate
    follows arithmetic BM with integrated variance xi_0^{i,e} * T_i, so
    S_{T_i} ~ N(S_0, integrated_var).

    Price = E^{A^{i,e}}[(K - S_{T_i})^+]
          = (K - S) * Phi(d) + sigma_T * phi(d)
    where d = (K - S) / sigma_T,  sigma_T = sqrt(integrated_var).

    Args:
        S: Current forward swap rate.
        K: Strike rate.
        integrated_var: xi_0 * T (total normal variance = xi_0^{i,e} * T_i).

    Returns:
        Expected payoff (annuity-normalized swaption price).
    """
    if integrated_var <= 0:
        return max(K - S, 0.0)
    sigma_T = np.sqrt(integrated_var)
    d = (K - S) / sigma_T
    # Bachelier receiver swaption: (K-S)*Phi(d) + sigma_T*phi(d)
    price = (K - S) * norm.cdf(d) + sigma_T * norm.pdf(d)
    return float(price)


def bs_receiver_swaption_vec(
    S: np.ndarray,
    K: float,
    integrated_var: float,
) -> np.ndarray:
    """Vectorized Bachelier receiver swaption price.

    Args:
        S: Array of forward swap rates.
        K: Strike rate (scalar).
        integrated_var: Total normal variance xi_0 * T.

    Returns:
        Array of expected payoffs.
    """
    S = np.asarray(S, dtype=float)
    if integrated_var <= 0:
        return np.maximum(K - S, 0.0)
    sigma_T = np.sqrt(integrated_var)
    d = (K - S) / sigma_T
    prices = (K - S) * norm.cdf(d) + sigma_T * norm.pdf(d)
    return prices
