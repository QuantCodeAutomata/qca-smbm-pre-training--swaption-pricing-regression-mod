"""Bootstrap formula for discount factors and annuity factors.

Recursive bootstrap from coterminal swap rates S^{u,e} to relative
discount factors P^u/P^e and relative annuity factors A_bar^{u,e} = A^{u,e}/P^e.
"""
import numpy as np
from typing import Tuple, Optional


def bootstrap_single(
    S_vec: np.ndarray,
    delta: float = 1.0,
    first_idx: int = 0,
    e: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap relative discount factors and annuity factors from coterminal swaps.
    
    Given coterminal swap rates S^{u,e} for u = first_idx, ..., e-1 (0-indexed),
    compute P^u/P^e and A_bar^{u,e} = A^{u,e}/P^e for u = first_idx, ..., e.
    
    Recursive formula (from u = e-1 down to first_idx):
      A_bar^{e-1,e} = delta
      P^{e-1}/P^e = 1 + S^{e-1,e} * A_bar^{e-1,e}
      For u = e-2,...,first_idx:
        A_bar^{u,e} = A_bar^{u+1,e} + P^{u+1}/P^e
        P^u/P^e = 1 + S^{u,e} * A_bar^{u,e}
    Also P^e/P^e = 1.
    
    Args:
        S_vec: Array of active swap rates, shape (n_active,),
               corresponding to S^{first_idx,e}, ..., S^{e-1,e}.
        delta: Tenor spacing (1.0 for annual payments).
        first_idx: Index of first active swap rate (0-based).
        e: Total number of components (5 by default, so T_e = T_6).
        
    Returns:
        P_bar: Relative discount factors P^u/P^e for u=first_idx,...,e, shape (n_active+1,).
               P_bar[0] = P^{first_idx}/P^e, ..., P_bar[n_active] = P^e/P^e = 1.
        A_bar: Relative annuity factors A_bar^{u,e} for u=first_idx,...,e-1, shape (n_active,).
               A_bar[0] = A_bar^{first_idx,e}, ..., A_bar[-1] = A_bar^{e-1,e} = delta.
    """
    n_active = len(S_vec)  # = e - first_idx
    
    # Arrays indexed from first_idx to e (inclusive for P, exclusive for A)
    P_bar = np.zeros(n_active + 1)  # P^{first_idx}/P^e, ..., P^e/P^e=1
    A_bar = np.zeros(n_active)      # A_bar^{first_idx,e}, ..., A_bar^{e-1,e}
    
    # P^e/P^e = 1 (terminal condition)
    P_bar[n_active] = 1.0
    
    # A_bar^{e-1,e} = delta (annual swap, one period before maturity)
    A_bar[n_active - 1] = delta
    # P^{e-1}/P^e = 1 + S^{e-1,e} * A_bar^{e-1,e}
    P_bar[n_active - 1] = 1.0 + S_vec[n_active - 1] * A_bar[n_active - 1]
    
    # Recursion downward
    for k in range(n_active - 2, -1, -1):
        # A_bar^{u,e} = A_bar^{u+1,e} + P^{u+1}/P^e
        A_bar[k] = A_bar[k + 1] + P_bar[k + 1]
        # P^u/P^e = 1 + S^{u,e} * A_bar^{u,e}
        P_bar[k] = 1.0 + S_vec[k] * A_bar[k]
    
    return P_bar, A_bar


def bootstrap_batch(
    S_batch: np.ndarray,
    delta: float = 1.0,
    first_idx: int = 0,
    e: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized bootstrap for a batch of paths.
    
    Args:
        S_batch: Swap rates array, shape (N_paths, n_active).
        delta: Tenor spacing.
        first_idx: First active component index.
        e: Total number of components.
        
    Returns:
        P_bar: shape (N_paths, n_active+1).
        A_bar: shape (N_paths, n_active).
    """
    N = S_batch.shape[0]
    n_active = S_batch.shape[1]
    
    P_bar = np.zeros((N, n_active + 1))
    A_bar = np.zeros((N, n_active))
    
    P_bar[:, n_active] = 1.0
    A_bar[:, n_active - 1] = delta
    P_bar[:, n_active - 1] = 1.0 + S_batch[:, n_active - 1] * A_bar[:, n_active - 1]
    
    for k in range(n_active - 2, -1, -1):
        A_bar[:, k] = A_bar[:, k + 1] + P_bar[:, k + 1]
        P_bar[:, k] = 1.0 + S_batch[:, k] * A_bar[:, k]
    
    return P_bar, A_bar


def log_abar_jacobian(
    S_vec: np.ndarray,
    i: int,
    delta: float = 1.0,
    first_idx: int = 0,
    h_scale: float = 1e-5,
) -> np.ndarray:
    """Compute d(ln A_bar^{i,e}) / d(S^j) for all active j via finite differences.
    
    Args:
        S_vec: Active swap rates, shape (n_active,).
        i: Index (relative to first_idx) of the annuity factor to differentiate.
        delta: Tenor spacing.
        first_idx: First active component index.
        h_scale: Finite difference step relative scale.
        
    Returns:
        Jacobian vector, shape (n_active,).
    """
    n_active = len(S_vec)
    _, A_bar_0 = bootstrap_single(S_vec, delta, first_idx)
    A_bar_i_0 = A_bar_0[i]
    
    if A_bar_i_0 <= 0:
        return np.zeros(n_active)
    
    jac = np.zeros(n_active)
    for j in range(n_active):
        h = h_scale * (abs(S_vec[j]) + 1e-6)
        S_perturbed = S_vec.copy()
        S_perturbed[j] += h
        _, A_bar_p = bootstrap_single(S_perturbed, delta, first_idx)
        A_bar_i_p = A_bar_p[i]
        if A_bar_i_p <= 0:
            jac[j] = 0.0
        else:
            jac[j] = (np.log(A_bar_i_p) - np.log(A_bar_i_0)) / h
    
    return jac
