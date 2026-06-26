"""Exp-1: SMBM Swaption Pricing Regression Models.

Pre-trains regression models v^{i,e,k}(S, X, tau) approximating
E^{A^{i,e}}[(k - S_{T_i}^{i,e})^+] for each coterminal pair i and strike k.

Using scipy.interpolate.RegularGridInterpolator - Context7 confirmed.
"""
import numpy as np
import pickle
import os
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Dict, Optional

# Import model parameters
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.params import (
    OMEGA, KAPPA, XI0, S0, T_EXERCISE, DELTA_T, RHO_SW,
    N_GRID, N_INNER, STRIKES
)
from src.black_scholes import bs_receiver_swaption


def sigma_t(
    xi0_i: float,
    X_t: np.ndarray,
    t: float,
    omega: float = OMEGA,
) -> np.ndarray:
    """Compute instantaneous vol sigma_t^{i,e} under annuity measure.
    
    sigma_t = sqrt(xi_0^{i,e}) * exp[(omega/2)*X_t - (omega^2/4)*t]
    
    Args:
        xi0_i: Initial variance xi_0^{i,e}.
        X_t: Variance state array, shape (N,).
        t: Current time.
        omega: Vol-of-vol parameter.
        
    Returns:
        Instantaneous vol array, shape (N,).
    """
    return np.sqrt(xi0_i) * np.exp((omega / 2.0) * X_t - (omega**2 / 4.0) * t)


def simulate_smbm_annuity(
    S0_grid: float,
    X0_grid: float,
    tau: float,
    xi0_i: float,
    omega: float = OMEGA,
    rho: float = RHO_SW,
    N_inner: int = N_INNER,
    delta_t: float = DELTA_T,
    seed: Optional[int] = None,
) -> float:
    """Simulate SMBM under annuity measure A^{i,e} and compute E[(k-S_tau)^+].
    
    Not yet parameterized for strike - returns terminal S values.
    Used to estimate swaption payoffs for multiple strikes simultaneously.
    
    Dynamics under A^{i,e} (kappa=0, no drift for S and X):
      dS_t = sigma_t * dW_t
      dX_t = dZ_t
    Correlated BMs: dW = L[0,0]*dB1, dZ = L[1,0]*dB1 + L[1,1]*dB2
    where L is Cholesky of [[1, rho],[rho, 1]].
    
    Args:
        S0_grid: Initial swap rate.
        X0_grid: Initial variance state.
        tau: Time to maturity.
        xi0_i: Initial variance xi_0^{i,e}.
        omega: Vol-of-vol.
        rho: Correlation rho(W, Z) = -0.259.
        N_inner: Number of inner MC paths.
        delta_t: Time step.
        seed: Optional random seed.
        
    Returns:
        Array of terminal swap rates S_{tau}, shape (N_inner,).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Cholesky factor for 2x2 correlation matrix [[1, rho],[rho, 1]]
    # L = [[1, 0], [rho, sqrt(1-rho^2)]]
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho**2)]])
    
    n_steps = max(1, int(round(tau / delta_t)))
    dt = tau / n_steps  # actual dt used
    
    S = np.full(N_inner, S0_grid)
    X = np.full(N_inner, X0_grid)
    t = 0.0
    
    for _ in range(n_steps):
        eps = rng.standard_normal((2, N_inner))
        # Correlated increments
        dW = (L[0, 0] * eps[0]) * np.sqrt(dt)
        dZ = (L[1, 0] * eps[0] + L[1, 1] * eps[1]) * np.sqrt(dt)
        
        sig = sigma_t(xi0_i, X, t, omega)
        S = S + sig * dW
        X = X + dZ
        t += dt
    
    return S


def compute_swaption_price(
    S0_grid: float,
    X0_grid: float,
    tau: float,
    strike: float,
    xi0_i: float,
    omega: float = OMEGA,
    rho: float = RHO_SW,
    N_inner: int = N_INNER,
    delta_t: float = DELTA_T,
    seed: Optional[int] = None,
) -> float:
    """Estimate E^{A^{i,e}}[(k - S_{tau})^+] by inner Monte Carlo.
    
    Args:
        S0_grid: Initial swap rate for this grid point.
        X0_grid: Initial variance state for this grid point.
        tau: Time to maturity.
        strike: Swaption strike k.
        xi0_i: Initial variance xi_0^{i,e}.
        omega: Vol-of-vol.
        rho: Within-factor correlation.
        N_inner: Inner MC paths.
        delta_t: Time step for simulation.
        seed: Optional random seed.
        
    Returns:
        Estimated swaption price (annuity-normalized).
    """
    if tau < 1e-6:
        return max(strike - S0_grid, 0.0)
    
    S_terminal = simulate_smbm_annuity(
        S0_grid, X0_grid, tau, xi0_i, omega, rho, N_inner, delta_t, seed
    )
    payoffs = np.maximum(strike - S_terminal, 0.0)
    return float(np.mean(payoffs))


def compute_swaption_grid(
    component_idx: int,
    strike: float,
    N_grid: int = N_GRID,
    N_inner: int = N_INNER,
    delta_t: float = DELTA_T,
    seed_base: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute swaption prices on the 3D grid (S, X, tau) for component i and strike k.
    
    Grid definition:
    - S in [max(0.001, S0 - 5*sigma_i), S0 + 5*sigma_i] with N_grid nodes
      where sigma_i = sqrt(xi0[i] * T_i)
    - X in [-4*sqrt(T_i), 4*sqrt(T_i)] with N_grid nodes
    - tau in [delta_t, T_i] with N_grid nodes
    
    Args:
        component_idx: Index i in {0,1,2,3,4} (0-indexed, corresponds to i=1,...,5).
        strike: Swaption strike k in {0.01, 0.015, 0.02}.
        N_grid: Number of grid nodes per dimension.
        N_inner: Inner MC paths per grid point.
        delta_t: Time step.
        seed_base: Base random seed.
        
    Returns:
        S_nodes: shape (N_grid,)
        X_nodes: shape (N_grid,)
        tau_nodes: shape (N_grid,)
        V_grid: shape (N_grid, N_grid, N_grid)
    """
    i = component_idx
    xi0_i = XI0[i]
    T_i = T_EXERCISE[i]
    
    # Grid boundaries
    sigma_i = np.sqrt(xi0_i * T_i)  # ATM total vol
    S_min = max(0.001, S0 - 5.0 * sigma_i)
    S_max = S0 + 5.0 * sigma_i
    X_min = -4.0 * np.sqrt(T_i)
    X_max = 4.0 * np.sqrt(T_i)
    tau_min = delta_t
    tau_max = T_i
    
    S_nodes = np.linspace(S_min, S_max, N_grid)
    X_nodes = np.linspace(X_min, X_max, N_grid)
    tau_nodes = np.linspace(tau_min, tau_max, N_grid)
    
    V_grid = np.zeros((N_grid, N_grid, N_grid))
    
    total_points = N_grid ** 3
    count = 0
    for si, S_val in enumerate(S_nodes):
        for xi, X_val in enumerate(X_nodes):
            for ti, tau_val in enumerate(tau_nodes):
                seed = seed_base + count
                price = compute_swaption_price(
                    S0_grid=S_val,
                    X0_grid=X_val,
                    tau=tau_val,
                    strike=strike,
                    xi0_i=xi0_i,
                    N_inner=N_inner,
                    seed=seed,
                )
                V_grid[si, xi, ti] = price
                count += 1
                if count % max(1, total_points // 10) == 0:
                    pct = 100 * count / total_points
                    print(f"  Grid progress: {pct:.0f}% ({count}/{total_points})")
    
    return S_nodes, X_nodes, tau_nodes, V_grid


def fit_swaption_model(
    S_nodes: np.ndarray,
    X_nodes: np.ndarray,
    tau_nodes: np.ndarray,
    V_grid: np.ndarray,
) -> RegularGridInterpolator:
    """Fit a RegularGridInterpolator to the swaption price grid.
    
    Using scipy.interpolate.RegularGridInterpolator with linear method.
    At boundary extrapolation, linear extrapolation is used (fill_value=None).
    
    Args:
        S_nodes: Grid nodes in S dimension, shape (N_grid,).
        X_nodes: Grid nodes in X dimension, shape (N_grid,).
        tau_nodes: Grid nodes in tau dimension, shape (N_grid,).
        V_grid: Swaption prices on grid, shape (N_grid, N_grid, N_grid).
        
    Returns:
        Fitted RegularGridInterpolator.
    """
    # Using RegularGridInterpolator - Context7 confirmed (scipy.interpolate)
    interp = RegularGridInterpolator(
        points=(S_nodes, X_nodes, tau_nodes),
        values=V_grid,
        method='linear',
        bounds_error=False,
        fill_value=None,  # extrapolate
    )
    return interp


def train_swaption_models(
    save_dir: str = 'data',
    N_grid: int = N_GRID,
    N_inner: int = N_INNER,
    components: Optional[list] = None,
    strikes: Optional[list] = None,
    verbose: bool = True,
) -> Dict[Tuple[int, float], dict]:
    """Train all 15 swaption pricing regression models.
    
    Trains models for each (component_idx i, strike k) pair.
    Saves each model as a pickle file.
    
    Args:
        save_dir: Directory to save models.
        N_grid: Grid resolution per dimension.
        N_inner: Inner Monte Carlo paths.
        components: List of component indices to train (default: all 5).
        strikes: List of strikes (default: all 3).
        verbose: Print progress messages.
        
    Returns:
        Dictionary mapping (i, k) -> {'model': interpolator, 'nodes': (S, X, tau), 'grid': V}.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if components is None:
        components = list(range(5))
    if strikes is None:
        strikes = STRIKES
    
    models = {}
    
    for i in components:
        for k in strikes:
            if verbose:
                print(f"\nTraining model i={i+1}, k={k:.3f} (T_i={T_EXERCISE[i]:.0f}y)")
            
            fname = os.path.join(save_dir, f'swaption_model_i{i+1}_k{k:.3f}.pkl')
            
            if os.path.exists(fname):
                if verbose:
                    print(f"  Loading cached model from {fname}")
                with open(fname, 'rb') as f:
                    model_data = pickle.load(f)
                models[(i, k)] = model_data
                continue
            
            S_nodes, X_nodes, tau_nodes, V_grid = compute_swaption_grid(
                component_idx=i,
                strike=k,
                N_grid=N_grid,
                N_inner=N_inner,
            )
            
            interp = fit_swaption_model(S_nodes, X_nodes, tau_nodes, V_grid)
            
            model_data = {
                'model': interp,
                'S_nodes': S_nodes,
                'X_nodes': X_nodes,
                'tau_nodes': tau_nodes,
                'V_grid': V_grid,
                'component_idx': i,
                'strike': k,
                'xi0_i': XI0[i],
                'T_i': T_EXERCISE[i],
            }
            
            with open(fname, 'wb') as f:
                pickle.dump(model_data, f)
            
            models[(i, k)] = model_data
            if verbose:
                print(f"  Saved to {fname}")
    
    return models


def load_swaption_models(save_dir: str = 'data') -> Dict[Tuple[int, float], dict]:
    """Load all 15 pre-trained swaption pricing models.
    
    Args:
        save_dir: Directory containing saved model pickle files.
        
    Returns:
        Dictionary mapping (i, k) -> model_data dict.
    """
    models = {}
    for i in range(5):
        for k in STRIKES:
            fname = os.path.join(save_dir, f'swaption_model_i{i+1}_k{k:.3f}.pkl')
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Model not found: {fname}. Run train_swaption_models first.")
            with open(fname, 'rb') as f:
                model_data = pickle.load(f)
            models[(i, k)] = model_data
    return models


def price_swaption(
    model_data: dict,
    S_t: float,
    X_t: float,
    tau: float,
) -> float:
    """Query the pre-trained regression model for swaption price.
    
    Args:
        model_data: Dict containing 'model' (RegularGridInterpolator) and grid info.
        S_t: Current swap rate.
        X_t: Current variance state.
        tau: Time to maturity.
        
    Returns:
        Annuity-normalized swaption price (clipped to non-negative).
    """
    if tau <= 1e-6:
        return max(model_data['strike'] - S_t, 0.0)
    
    point = np.array([[S_t, X_t, tau]])
    price = float(model_data['model'](point)[0])
    return max(price, 0.0)  # non-negativity


def price_swaption_batch(
    model_data: dict,
    S_batch: np.ndarray,
    X_batch: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Query pre-trained model for a batch of paths.
    
    Args:
        model_data: Model dict.
        S_batch: Swap rates, shape (N_paths,).
        X_batch: Variance states, shape (N_paths,).
        tau: Time to maturity (scalar).
        
    Returns:
        Swaption prices, shape (N_paths,), clipped to >= 0.
    """
    if tau <= 1e-6:
        return np.maximum(model_data['strike'] - S_batch, 0.0)
    
    tau_arr = np.full_like(S_batch, tau)
    points = np.stack([S_batch, X_batch, tau_arr], axis=1)
    prices = model_data['model'](points)
    return np.maximum(prices, 0.0)


def validate_swaption_model(
    model_data: dict,
    test_fraction: float = 0.2,
    N_inner_val: int = N_INNER,
    seed_offset: int = 9999,
) -> dict:
    """Validate swaption regression model on held-out grid points.
    
    Args:
        model_data: Trained model data dict.
        test_fraction: Fraction of grid points for validation.
        N_inner_val: Inner MC paths for validation ground truth.
        seed_offset: Random seed offset for validation MC.
        
    Returns:
        Validation results dict with relative errors and stats.
    """
    i = model_data['component_idx']
    k = model_data['strike']
    xi0_i = model_data['xi0_i']
    S_nodes = model_data['S_nodes']
    X_nodes = model_data['X_nodes']
    tau_nodes = model_data['tau_nodes']
    V_grid = model_data['V_grid']
    
    N_grid = len(S_nodes)
    N_total = N_grid ** 3
    N_test = int(N_total * test_fraction)
    
    rng = np.random.default_rng(seed_offset)
    # Sample test indices
    all_indices = [(si, xi, ti) for si in range(N_grid)
                   for xi in range(N_grid) for ti in range(N_grid)]
    test_mask = rng.choice(len(all_indices), size=N_test, replace=False)
    test_indices = [all_indices[j] for j in test_mask]
    
    rel_errors = []
    abs_errors = []
    
    for si, xi, ti in test_indices:
        S_val = S_nodes[si]
        X_val = X_nodes[xi]
        tau_val = tau_nodes[ti]
        v_true = V_grid[si, xi, ti]
        
        v_pred = price_swaption(model_data, S_val, X_val, tau_val)
        
        abs_err = abs(v_pred - v_true)
        if v_true > 1e-6:
            rel_err = abs_err / v_true
            rel_errors.append(rel_err)
        abs_errors.append(abs_err)
    
    results = {
        'n_test_points': N_test,
        'mean_rel_error': np.mean(rel_errors) if rel_errors else np.nan,
        'max_rel_error': np.max(rel_errors) if rel_errors else np.nan,
        'mean_abs_error': np.mean(abs_errors),
        'max_abs_error': np.max(abs_errors),
        'pct_within_01pct': np.mean(np.array(rel_errors) < 0.001) if rel_errors else np.nan,
    }
    return results


def validate_bs_consistency(models: dict) -> dict:
    """Check ATM swaption prices match Black-Scholes at (S=k=0.02, X=0, tau=T_i).
    
    Args:
        models: Dict of all trained models.
        
    Returns:
        Dict with BS prices, model prices, and relative errors for each component.
    """
    results = {}
    for i in range(5):
        k = 0.02
        T_i = T_EXERCISE[i]
        xi0_i = XI0[i]
        
        bs_price = bs_receiver_swaption(S=0.02, K=0.02, integrated_var=xi0_i * T_i)
        
        model_data = models[(i, k)]
        model_price = price_swaption(model_data, S_t=0.02, X_t=0.0, tau=T_i)
        
        rel_err = abs(model_price - bs_price) / max(bs_price, 1e-10)
        
        results[i] = {
            'T_i': T_i,
            'xi0_i': xi0_i,
            'bs_price': bs_price,
            'model_price': model_price,
            'rel_error': rel_err,
            'bs_price_bp': bs_price * 10000,
            'model_price_bp': model_price * 10000,
        }
    
    return results
