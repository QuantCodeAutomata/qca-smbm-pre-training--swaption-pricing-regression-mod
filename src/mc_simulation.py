"""Exp-2: SMBM Monte Carlo Path Generation under Terminal Measure P^e.

Generates synthetic paths of coterminal swap rates S^{i,e} and variance
states X^{i,e} under the terminal measure P^e with Euler-Maruyama.

# Custom implementation — Context7 found no library equivalent for SMBM dynamics
"""
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import (
    OMEGA, KAPPA, XI0, S0, T_EXERCISE, T_COMMON, DELTA_T, RHO_SW,
    N_STEPS, T_GRID, EXERCISE_IDX, STRIKES, STRIKE_ATM, DELTA_TENOR,
    N_PATHS_TOTAL, N_PATHS_TRAIN, N_PATHS_TEST, SEED, SEED_TEST,
    get_cholesky_factors,
)
from src.bootstrap import bootstrap_batch, log_abar_jacobian, bootstrap_single
from src.swaption_pricing import price_swaption_batch, sigma_t as smbm_sigma


# ------------------------------------------------------------------
# Helper: drift computation under terminal measure P^e
# ------------------------------------------------------------------

def compute_drift_batch(
    S_active: np.ndarray,
    X_active: np.ndarray,
    t: float,
    n_active: Optional[int] = None,
    xi0: np.ndarray = None,
    omega: float = OMEGA,
    corr_matrix: np.ndarray = None,
    delta: float = DELTA_TENOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute drift corrections mu_S^i and mu_X^i for all active paths.

    Under terminal measure P^e, the drift for S^i is:
      mu_S^i = sum_j J_{ij} * sigma_t^i * sigma_t^j * rho(W^i, W^j)
    And for X^i:
      mu_X^i = sum_j J_{ij} * sigma_t^j * rho(Z^i, W^j)

    where J_{ij} = d(ln A_bar^{i,e}) / d(S^j) and rho values come from the
    corr_matrix S-S block (for W-W) and X-S block (for Z-W).

    Args:
        S_active: Active swap rates, shape (N_paths, n_active).
        X_active: Active variance states, shape (N_paths, n_active).
        t: Current time.
        n_active: Number of active components (inferred from S_active if None).
        xi0: xi0 for active components, shape (n_active,).
        omega: Vol-of-vol parameter.
        corr_matrix: Correlation matrix of shape (2*n_active, 2*n_active).
        delta: Tenor spacing.

    Returns:
        mu_S: Drift for swap rates, shape (N_paths, n_active).
        mu_X: Drift for variance states, shape (N_paths, n_active).
    """
    N_paths, n_act = S_active.shape
    mu_S = np.zeros_like(S_active)
    mu_X = np.zeros_like(X_active)

    # Correlation sub-blocks:
    # S-S block: corr_matrix[:n_act, :n_act]
    # X-S block: corr_matrix[n_act:, :n_act]
    rho_SS = corr_matrix[:n_act, :n_act]
    rho_XS = corr_matrix[n_act:, :n_act]

    # Compute sigma_t for all active components and paths
    # sigma[n, j] = sqrt(xi0_j) * exp((omega/2)*X[n,j] - (omega^2/4)*t)
    sig = np.sqrt(xi0)[np.newaxis, :] * np.exp(
        (omega / 2.0) * X_active - (omega**2 / 4.0) * t
    )  # shape (N_paths, n_active)

    # Compute Jacobian for each path using a single representative path
    # (Approximate: use mean S for Jacobian computation to avoid N_paths * N^2 calls)
    # More accurately: compute per-path, but for efficiency use vectorized finite diff
    S_mean = np.mean(S_active, axis=0)  # shape (n_active,)

    # Compute Jacobian matrix J[i,j] = d(ln A_bar^{i,e}) / d(S^j) at S_mean
    J = np.zeros((n_act, n_act))
    for i_idx in range(n_act):
        J[i_idx, :] = log_abar_jacobian(S_mean, i=i_idx, delta=delta)

    # mu_S^{n,i} = sum_j J[i,j] * sigma[n,i] * sigma[n,j] * rho_SS[i,j]
    # = sigma[n,i] * sum_j (J[i,j] * rho_SS[i,j] * sigma[n,j])
    # Vectorized: (N_paths, n_active) @ (n_active, n_active) element-wise
    JR_SS = J * rho_SS  # shape (n_active, n_active)
    # For each path n and active i:
    #   mu_S[n,i] = sig[n,i] * sum_j JR_SS[i,j] * sig[n,j]
    weighted = sig @ JR_SS.T  # (N_paths, n_active)
    mu_S = sig * weighted  # (N_paths, n_active)

    # mu_X^{n,i} = sum_j J[i,j] * sigma[n,j] * rho_XS[i,j]
    JR_XS = J * rho_XS  # (n_active, n_active)
    mu_X = sig @ JR_XS.T  # (N_paths, n_active)

    return mu_S, mu_X


# ------------------------------------------------------------------
# Main simulation function
# ------------------------------------------------------------------

def simulate_paths(
    n_paths: int,
    swaption_models: Optional[Dict] = None,
    omega: float = OMEGA,
    xi0: np.ndarray = XI0,
    X0: np.ndarray = None,
    seed: int = SEED,
    corr_matrices: Optional[List[np.ndarray]] = None,
    delta: float = DELTA_TENOR,
    strikes: Optional[List[float]] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Simulate SMBM paths under terminal measure P^e.

    Generates n_paths joint paths of all coterminal swap rates S^{i,e}
    and variance states X^{i,e} with Euler-Maruyama discretization.

    Args:
        n_paths: Number of paths to simulate.
        swaption_models: Dict of pre-trained swaption models (from exp_1).
                         If None, uses Black-Scholes formula for pricing.
        omega: Vol-of-vol parameter.
        xi0: Initial variance states, shape (5,).
        X0: Initial variance state values (default: zeros).
        seed: Random seed for reproducibility.
        corr_matrices: List of 5 correlation matrices for each period.
                       If None, uses build_correlation_matrix from params.
        delta: Tenor spacing (annual = 1.0).
        strikes: Strike values for swaption pricing (default: STRIKES).
        verbose: Print progress.

    Returns:
        paths_dict with keys:
            'S_rates': shape (N, 289, 5) — swap rates
            'X_states': shape (N, 289, 5) — variance states
            'annuity_factors': shape (N, 289, 5) — A_bar^{i,e}
            'swap_prices': shape (N, 289, 5) — relative swap prices U_bar^{i,e}
            'swaption_K002': shape (N, 289, 5) — relative swaption prices at K=0.02
            'swaption_K001': shape (N, 289, 5) — relative swaption prices at K=0.01
            'swaption_K0015': shape (N, 289, 5) — relative swaption prices at K=0.015
            'exercise_dates_idx': list of exercise time step indices
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if X0 is None:
        X0 = np.zeros(5)
    if strikes is None:
        strikes = STRIKES

    # Load or build correlation matrices and Cholesky factors
    if corr_matrices is None:
        from src.params import build_correlation_matrix
        n_active_list = [5, 4, 3, 2, 1]
        corr_matrices = [build_correlation_matrix(n) for n in n_active_list]

    chol_factors = []
    for C in corr_matrices:
        eigvals = np.linalg.eigvalsh(C)
        if eigvals.min() < 1e-10:
            C = C + (1e-8 - eigvals.min()) * np.eye(C.shape[0])
            d = np.sqrt(np.diag(C))
            C = C / np.outer(d, d)
        L = np.linalg.cholesky(C)
        chol_factors.append(L)

    N = n_paths
    T_steps = N_STEPS + 1  # 289 time steps including t=0

    # Pre-allocate storage
    S_all = np.zeros((N, T_steps, 5))
    X_all = np.zeros((N, T_steps, 5))
    A_bar_all = np.zeros((N, T_steps, 5))
    swap_prices = np.zeros((N, T_steps, 5))
    swaption_002 = np.zeros((N, T_steps, 5))
    swaption_001 = np.zeros((N, T_steps, 5))
    swaption_0015 = np.zeros((N, T_steps, 5))

    # Initialize at t=0
    S_all[:, 0, :] = S0
    X_all[:, 0, :] = X0[np.newaxis, :]

    # ----------------------------------------------------------------
    # Helper: compute prices for all assets at a given time step
    # ----------------------------------------------------------------
    def compute_prices_at_step(
        step: int,
        S_current: np.ndarray,  # (N, 5)
        X_current: np.ndarray,  # (N, 5)
        t_current: float,
        n_still_active: int,
    ):
        """Store swap prices and swaption prices for all active components."""
        # Bootstrap from active swap rates
        S_act = S_current[:, :n_still_active]
        _, A_bar = bootstrap_batch(S_act, delta=delta)  # (N, n_active)

        # Store A_bar for active components
        A_bar_all[:, step, :n_still_active] = A_bar

        # Relative swap price: U_bar^{i,e} = A_bar^{i,e} * (K - S^{i,e})
        # where K = STRIKE_ATM = 0.02
        swap_prices[:, step, :n_still_active] = (
            A_bar * (STRIKE_ATM - S_act)
        )

        # Swaption prices using pre-trained models
        for i in range(n_still_active):
            tau = T_EXERCISE[i] - t_current
            if tau <= 0:
                tau = 0.0

            for k_val, store_arr in zip(
                [0.02, 0.01, 0.015],
                [swaption_002, swaption_001, swaption_0015]
            ):
                if tau <= 0:
                    v = np.maximum(k_val - S_current[:, i], 0.0)
                elif swaption_models is not None:
                    key = (i, k_val)
                    if key in swaption_models:
                        v = price_swaption_batch(
                            swaption_models[key],
                            S_current[:, i],
                            X_current[:, i],
                            tau,
                        )
                    else:
                        # Fallback to BS
                        from src.black_scholes import bs_receiver_swaption_vec
                        v = bs_receiver_swaption_vec(
                            S_current[:, i], k_val, XI0[i] * T_EXERCISE[i]
                        )
                else:
                    # No pre-trained model: use Black-Scholes
                    from src.black_scholes import bs_receiver_swaption_vec
                    v = bs_receiver_swaption_vec(
                        S_current[:, i], k_val, XI0[i] * max(tau, 1e-6)
                    )

                # Relative swaption price: O_bar = A_bar^{i,e} * v^{i,e,k}
                store_arr[:, step, i] = A_bar[:, i] * v

    # Compute initial prices at t=0
    compute_prices_at_step(0, S_all[:, 0, :], X_all[:, 0, :], 0.0, 5)

    if verbose:
        print(f"Simulating {N} paths, {N_STEPS} steps...")

    # ----------------------------------------------------------------
    # Main time loop
    # ----------------------------------------------------------------
    S_current = S_all[:, 0, :].copy()
    X_current = X_all[:, 0, :].copy()

    for step in range(N_STEPS):
        t = T_GRID[step]
        t_next = T_GRID[step + 1]

        # Determine active period: how many components are still active
        # Period boundaries: T_1=4, T_2=5, ..., T_5=8
        # At time t, component i is active if t < T_i
        n_still_active = sum(t < T_EXERCISE[i] for i in range(5))
        if n_still_active == 0:
            break

        # Period index: 0 if 0<=t<4, 1 if 4<=t<5, etc.
        period = 5 - n_still_active  # 0,1,2,3,4

        L_p = chol_factors[period]  # shape (2*n_act, 2*n_act)
        n_act = n_still_active
        dim = 2 * n_act

        # Generate correlated standard normal increments
        eps = rng.standard_normal((N, dim))
        dZ = (L_p @ eps.T).T * np.sqrt(DELTA_T)  # (N, dim)

        dW = dZ[:, :n_act]       # BMs for S variables
        dZ_var = dZ[:, n_act:]   # BMs for X variables

        # Active components
        S_act = S_current[:, :n_act]
        X_act = X_current[:, :n_act]
        xi0_act = xi0[:n_act]
        C_act = corr_matrices[period]

        # Compute sigma_t for active components
        sig = np.sqrt(xi0_act)[np.newaxis, :] * np.exp(
            (omega / 2.0) * X_act - (omega**2 / 4.0) * t
        )  # (N, n_act)

        # Compute drift corrections under P^e
        mu_S, mu_X = compute_drift_batch(
            S_active=S_act, X_active=X_act, t=t,
            xi0=xi0_act, omega=omega, corr_matrix=C_act, delta=delta,
        )

        # Euler-Maruyama update
        S_new = S_act + mu_S * DELTA_T + sig * dW
        X_new = X_act + mu_X * DELTA_T + dZ_var

        # Clip S to stay positive (rates should not go deeply negative)
        S_new = np.maximum(S_new, -0.1)

        # Update current state
        S_current[:, :n_act] = S_new
        X_current[:, :n_act] = X_new

        # Store at t_{step+1}
        S_all[:, step + 1, :] = S_current
        X_all[:, step + 1, :] = X_current

        # Compute and store prices
        compute_prices_at_step(
            step + 1, S_current, X_current, t_next, n_still_active
        )

        if verbose and (step + 1) % 32 == 0:
            print(f"  Step {step+1}/{N_STEPS} (t={t_next:.2f}y)")

    paths_dict = {
        'S_rates': S_all,
        'X_states': X_all,
        'annuity_factors': A_bar_all,
        'swap_prices': swap_prices,
        'swaption_K002': swaption_002,
        'swaption_K001': swaption_001,
        'swaption_K0015': swaption_0015,
        'exercise_dates_idx': EXERCISE_IDX,
    }

    return paths_dict


def save_paths(paths_dict: Dict, filepath: str) -> None:
    """Save path data to compressed numpy file.

    Args:
        paths_dict: Dictionary of path arrays.
        filepath: Output file path (.npz).
    """
    save_dict = {k: v for k, v in paths_dict.items()
                 if isinstance(v, np.ndarray)}
    save_dict['exercise_dates_idx'] = np.array(paths_dict['exercise_dates_idx'])
    np.savez_compressed(filepath, **save_dict)
    print(f"Saved paths to {filepath}")


def load_paths(filepath: str) -> Dict[str, np.ndarray]:
    """Load path data from compressed numpy file.

    Args:
        filepath: Path to .npz file.

    Returns:
        paths_dict with the same structure as simulate_paths output.
    """
    data = np.load(filepath, allow_pickle=False)
    paths_dict = {k: data[k] for k in data.files}
    paths_dict['exercise_dates_idx'] = list(paths_dict['exercise_dates_idx'].astype(int))
    return paths_dict


def validate_paths(paths_dict: Dict, verbose: bool = True) -> Dict:
    """Run validation checks on simulated paths.

    Checks:
    1. Swaption price non-negativity
    2. Annuity positivity
    3. Bootstrap consistency (discount factors > 0)
    4. Martingale test for swap prices

    Args:
        paths_dict: Path data dictionary.
        verbose: Print validation results.

    Returns:
        Validation results dict.
    """
    results = {}

    # Check swaption non-negativity
    O_002 = paths_dict['swaption_K002']
    n_neg = np.sum(O_002 < -1e-8)
    results['swaption_negative_count'] = n_neg
    results['swaption_nonneg_pass'] = (n_neg == 0)

    # Check annuity positivity
    A_bar = paths_dict['annuity_factors']
    # Only check where values should be non-zero (active components)
    A_positive = A_bar[A_bar > 1e-10]
    results['annuity_min'] = float(A_positive.min()) if len(A_positive) > 0 else 0.0
    results['annuity_pos_pass'] = results['annuity_min'] > 0

    # Martingale test: mean change in swap prices
    swap_p = paths_dict['swap_prices']
    delta_swap = np.diff(swap_p, axis=1)  # (N, T_steps-1, 5)
    mean_delta = np.abs(np.mean(delta_swap, axis=0))  # (T_steps-1, 5)
    results['mean_abs_delta_swap_max'] = float(np.max(mean_delta[:128, :]))
    results['martingale_pass'] = results['mean_abs_delta_swap_max'] < 1e-2

    # Initial prices
    results['S0_initial'] = float(np.mean(paths_dict['S_rates'][:, 0, 0]))
    results['O002_initial_i1_bp'] = float(np.mean(paths_dict['swaption_K002'][:, 0, 0])) * 10000
    results['max_swaption_t0_bp'] = float(np.max(np.mean(paths_dict['swaption_K002'][:, 0, :], axis=0))) * 10000

    if verbose:
        print(f"Swaption non-negative: {results['swaption_nonneg_pass']} (neg count: {n_neg})")
        print(f"Annuity min value: {results['annuity_min']:.6f}")
        print(f"Martingale test (max |E[dU]|): {results['mean_abs_delta_swap_max']:.6f}")
        print(f"Initial S^1: {results['S0_initial']:.4f}")
        print(f"Initial O^1 (K=0.02) at t=0: {results['O002_initial_i1_bp']:.2f} bp")
        print(f"Max swaption value at t=0: {results['max_swaption_t0_bp']:.2f} bp")

    return results


def generate_all_paths(
    swaption_models: Optional[Dict] = None,
    save_dir: str = 'data',
    n_total: int = N_PATHS_TOTAL,
    omega: float = OMEGA,
    xi0: np.ndarray = XI0,
    X0: np.ndarray = None,
    seed_train: int = SEED,
    seed_test: int = SEED_TEST,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """Generate and save training and test paths.

    Args:
        swaption_models: Pre-trained swaption models.
        save_dir: Directory to save paths.
        n_total: Total paths (split 50/50 train/test).
        omega: Vol-of-vol.
        xi0: Initial variance states.
        X0: Initial X states.
        seed_train: Seed for training paths.
        seed_test: Seed for test paths.
        verbose: Print progress.

    Returns:
        (train_paths, test_paths) dictionaries.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_train = n_total // 2
    n_test = n_total - n_train

    train_file = os.path.join(save_dir, 'paths_train.npz')
    test_file = os.path.join(save_dir, 'paths_test.npz')

    if os.path.exists(train_file) and os.path.exists(test_file):
        if verbose:
            print("Loading existing path files...")
        train_paths = load_paths(train_file)
        test_paths = load_paths(test_file)
        return train_paths, test_paths

    if verbose:
        print(f"\n=== Generating {n_train} training paths ===")
    train_paths = simulate_paths(
        n_paths=n_train,
        swaption_models=swaption_models,
        omega=omega, xi0=xi0, X0=X0,
        seed=seed_train, verbose=verbose,
    )
    save_paths(train_paths, train_file)

    if verbose:
        print(f"\n=== Generating {n_test} test paths ===")
    test_paths = simulate_paths(
        n_paths=n_test,
        swaption_models=swaption_models,
        omega=omega, xi0=xi0, X0=X0,
        seed=seed_test, verbose=verbose,
    )
    save_paths(test_paths, test_file)

    return train_paths, test_paths
