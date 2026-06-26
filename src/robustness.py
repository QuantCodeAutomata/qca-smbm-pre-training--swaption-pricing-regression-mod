"""Exp-5: Cross-Parameter Robustness Validation.

Generates paths under modified SMBM parameters (halved vol-of-vol,
flipped rate-vol correlations) and evaluates all trained models without
retraining to assess out-of-sample robustness.

# Custom - Context7 found no library equivalent for SMBM parameter modification
"""
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import (
    OMEGA, XI0, T_EXERCISE, DELTA_T, SEED, SEED_TEST, N_PATHS_TRAIN,
    ALPHA_LEVELS, build_correlation_matrix,
)
from src.mc_simulation import simulate_paths, save_paths, load_paths
from src.evaluation import (
    compute_metrics, evaluate_all_configurations, save_tables,
    run_consistency_checks, compute_exercise_payoffs,
)


OMEGA_MOD: float = OMEGA / 2.0  # = 0.42015


def build_modified_correlation_matrices(
    original_matrices: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """Build modified correlation matrices with flipped S-X and X-S blocks.

    Sign-flips all rho(S^i, X^j) and rho(X^i, S^j) entries in each period's
    correlation matrix. S-S and X-X blocks are left unchanged.

    Args:
        original_matrices: Original correlation matrices (5 matrices).
                           If None, builds from default parameters.

    Returns:
        List of 5 modified correlation matrices, one per active period.
    """
    if original_matrices is None:
        n_active_list = [5, 4, 3, 2, 1]
        original_matrices = [build_correlation_matrix(n) for n in n_active_list]

    modified_matrices = []

    for C_orig in original_matrices:
        n = C_orig.shape[0] // 2  # n active components
        C_mod = C_orig.copy()

        # Negate S-X block (rows 0:n, cols n:2n)
        C_mod[:n, n:] = -C_orig[:n, n:]
        # Negate X-S block (rows n:2n, cols 0:n)
        C_mod[n:, :n] = -C_orig[n:, :n]

        # Ensure diagonal = 1
        np.fill_diagonal(C_mod, 1.0)

        # Verify positive semi-definiteness
        eigvals = np.linalg.eigvalsh(C_mod)
        if eigvals.min() < -1e-10:
            eps = max(1e-6, -eigvals.min() + 1e-7)
            C_mod += eps * np.eye(C_mod.shape[0])
            # Renormalize
            d = np.sqrt(np.diag(C_mod))
            C_mod = C_mod / np.outer(d, d)
            np.fill_diagonal(C_mod, 1.0)

        modified_matrices.append(C_mod)

    return modified_matrices


def verify_atm_price_consistency(
    omega_new: float = OMEGA_MOD,
    xi0: np.ndarray = XI0,
    n_verify: int = 1000,
    seed: int = 42,
) -> Dict[int, Dict]:
    """Verify ATM swaption prices unchanged under parameter modification.

    At (S=K=0.02, X=0, kappa=0), the ATM swaption price equals Black-Scholes
    with integrated variance xi_0^{i,e} * T_i, independent of omega.

    Args:
        omega_new: Modified vol-of-vol.
        xi0: Initial variance states.
        n_verify: Paths for verification MC.
        seed: Random seed.

    Returns:
        Dict with original and modified ATM prices for each component i.
    """
    from src.black_scholes import bs_receiver_swaption
    from src.swaption_pricing import simulate_smbm_annuity

    results = {}
    for i in range(5):
        T_i = T_EXERCISE[i]
        xi0_i = XI0[i]
        bs_price = bs_receiver_swaption(0.02, 0.02, xi0_i * T_i)

        # Verify with omega_new (marginal distribution of S_{T_i} is same)
        S_term_orig = simulate_smbm_annuity(
            0.02, 0.0, T_i, xi0_i,
            omega=OMEGA, N_inner=n_verify, seed=seed
        )
        S_term_mod = simulate_smbm_annuity(
            0.02, 0.0, T_i, xi0_i,
            omega=omega_new, N_inner=n_verify, seed=seed + 1000
        )
        price_orig = float(np.mean(np.maximum(0.02 - S_term_orig, 0.0)))
        price_mod = float(np.mean(np.maximum(0.02 - S_term_mod, 0.0)))

        results[i] = {
            'T_i': T_i,
            'bs_price_bp': bs_price * 10000,
            'mc_price_orig_bp': price_orig * 10000,
            'mc_price_mod_bp': price_mod * 10000,
            'diff_orig_bp': abs(price_orig - bs_price) * 10000,
            'diff_mod_bp': abs(price_mod - bs_price) * 10000,
        }
        print(f"  i={i+1}: BS={bs_price*1e4:.2f}bp, orig={price_orig*1e4:.2f}bp, "
              f"mod={price_mod*1e4:.2f}bp")

    return results


def generate_modified_paths(
    swaption_models: Optional[Dict] = None,
    save_dir: str = 'data',
    n_paths: int = None,
    seed: int = SEED_TEST + 100,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate paths under modified SMBM parameters.

    Uses omega_mod = 0.42015 and sign-flipped S-X correlations.
    X_0 = 0 is unchanged (ATM prices are omega-independent).

    Args:
        swaption_models: Pre-trained swaption models from exp_1.
        save_dir: Directory to save paths.
        n_paths: Number of paths (default: N_PATHS_TRAIN).
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Modified paths dictionary.
    """
    if n_paths is None:
        from src.params import N_PATHS_TOTAL
        n_paths = N_PATHS_TOTAL // 2  # same as test size

    os.makedirs(save_dir, exist_ok=True)
    mod_file = os.path.join(save_dir, 'paths_modified.npz')

    if os.path.exists(mod_file):
        if verbose:
            print(f"Loading existing modified paths from {mod_file}")
        return load_paths(mod_file)

    if verbose:
        print(f"\n=== Generating {n_paths} modified-parameter paths ===")
        print(f"  omega_mod = {OMEGA_MOD:.5f} (original = {OMEGA:.4f})")
        print("  S-X correlations: sign-flipped")
        print("  X_0 = 0 (unchanged)")

    # Build modified correlation matrices
    mod_corr_matrices = build_modified_correlation_matrices()

    # Simulate paths with modified parameters
    mod_paths = simulate_paths(
        n_paths=n_paths,
        swaption_models=swaption_models,
        omega=OMEGA_MOD,
        xi0=XI0,
        X0=np.zeros(5),
        seed=seed,
        corr_matrices=mod_corr_matrices,
        verbose=verbose,
    )

    save_paths(mod_paths, mod_file)

    return mod_paths


def evaluate_robustness(
    trained_models: Dict,
    paths_modified: Dict[str, np.ndarray],
    alpha_levels: Optional[List[float]] = None,
    basis_points_factor: float = 10000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all trained models on modified-parameter paths.

    Applies models trained on original parameters to paths generated
    under modified parameters (omega halved, correlations flipped).

    Args:
        trained_models: Dict mapping (strategy, alpha) -> BermudanHedger.
        paths_modified: Modified parameter path dictionary.
        alpha_levels: CVaR levels.
        basis_points_factor: Multiplier for basis points conversion.

    Returns:
        (table3, table4): DataFrames for Tables 3 and 4.
    """
    if alpha_levels is None:
        alpha_levels = ALPHA_LEVELS

    # Compute max swaption value at t=0 for modified paths
    opt002_t0 = paths_modified['swaption_K002'][:, 0, :]  # (N, 5)
    max_swaption_t0 = float(np.max(np.mean(opt002_t0, axis=0)))
    max_swaption_t0_bp = max_swaption_t0 * basis_points_factor
    print(f"Max swaption value at t=0 (modified): {max_swaption_t0_bp:.2f} bp")

    rows_t3 = []
    rows_t4 = []
    bp = basis_points_factor

    for (strategy, alpha), hedger in sorted(trained_models.items()):
        print(f"\nRobustness eval: {strategy}, alpha={alpha}...")

        # Evaluate on modified paths (no retraining)
        pl_array = hedger.evaluate(paths_modified)

        try:
            ex_payoffs = compute_exercise_payoffs(hedger, paths_modified)
        except Exception:
            ex_payoffs = None

        metrics = compute_metrics(
            pl_array, alpha, ex_payoffs, max_swaption_t0
        )

        rows_t3.append({
            'Strategy': strategy,
            'Alpha': alpha,
            'Model_Value_bp': metrics['model_value'] * bp,
            'NonArb_Value_bp': metrics['nonarb_value'] * bp,
            'Hedge_PnL_Mean_bp': metrics['hedge_pnl_mean'] * bp,
            'Model_Switch_bp': metrics['model_switch'] * bp,
            'NonArb_Switch_bp': metrics['nonarb_switch'] * bp,
        })

        rows_t4.append({
            'Strategy': strategy,
            'Alpha': alpha,
            'P25_bp': metrics['p25'] * bp,
            'P50_bp': metrics['p50'] * bp,
            'P75_bp': metrics['p75'] * bp,
            'IQR_bp': metrics['iqr'] * bp,
            'Loss_Prob': metrics['loss_prob'],
            'Expected_Loss_bp': metrics['expected_loss'] * bp,
            'CVaR95_bp': metrics['cvar95'] * bp,
            'CVaR99_bp': metrics['cvar99'] * bp,
        })

    # S^Max on modified paths
    from src.deep_hedging import smax_strategy
    opt002_mod = paths_modified['swaption_K002']
    S_rates_mod = paths_modified['S_rates']
    pl_smax_mod = smax_strategy(opt002_mod, list(paths_modified['exercise_dates_idx']), S_rates=S_rates_mod)

    for alpha in alpha_levels:
        metrics = compute_metrics(pl_smax_mod, alpha, None, max_swaption_t0)

        rows_t3.append({
            'Strategy': 'S_Max',
            'Alpha': alpha,
            'Model_Value_bp': metrics['model_value'] * bp,
            'NonArb_Value_bp': metrics['nonarb_value'] * bp,
            'Hedge_PnL_Mean_bp': metrics['hedge_pnl_mean'] * bp,
            'Model_Switch_bp': metrics['model_switch'] * bp,
            'NonArb_Switch_bp': metrics['nonarb_switch'] * bp,
        })

        rows_t4.append({
            'Strategy': 'S_Max',
            'Alpha': alpha,
            'P25_bp': metrics['p25'] * bp,
            'P50_bp': metrics['p50'] * bp,
            'P75_bp': metrics['p75'] * bp,
            'IQR_bp': metrics['iqr'] * bp,
            'Loss_Prob': metrics['loss_prob'],
            'Expected_Loss_bp': metrics['expected_loss'] * bp,
            'CVaR95_bp': metrics['cvar95'] * bp,
            'CVaR99_bp': metrics['cvar99'] * bp,
        })

    table3 = pd.DataFrame(rows_t3).sort_values(['Strategy', 'Alpha']).reset_index(drop=True)
    table4 = pd.DataFrame(rows_t4).sort_values(['Strategy', 'Alpha']).reset_index(drop=True)

    return table3, table4


def compute_robustness_summary(
    table1: pd.DataFrame,
    table3: pd.DataFrame,
    table2: pd.DataFrame,
    table4: pd.DataFrame,
) -> pd.DataFrame:
    """Compute degradation from original to modified parameter tables.

    Args:
        table1: Original scenario Table 1.
        table3: Modified scenario Table 3.
        table2: Original scenario Table 2.
        table4: Modified scenario Table 4.

    Returns:
        Summary DataFrame showing absolute changes for key metrics.
    """
    # Merge on Strategy + Alpha
    merged1 = table1.merge(table3, on=['Strategy', 'Alpha'], suffixes=('_orig', '_mod'))
    merged2 = table2.merge(table4, on=['Strategy', 'Alpha'], suffixes=('_orig', '_mod'))

    summary_rows = []
    for _, row in merged1.iterrows():
        strat = row['Strategy']
        alpha = row['Alpha']

        delta_mv = row['Model_Value_bp_mod'] - row['Model_Value_bp_orig']

        # Find IQR change
        iqr_orig = merged2.loc[
            (merged2['Strategy'] == strat) & (merged2['Alpha'] == alpha),
            'IQR_bp_orig'
        ].values
        iqr_mod = merged2.loc[
            (merged2['Strategy'] == strat) & (merged2['Alpha'] == alpha),
            'IQR_bp_mod'
        ].values

        delta_iqr = (iqr_mod[0] - iqr_orig[0]) if len(iqr_orig) > 0 else np.nan

        summary_rows.append({
            'Strategy': strat,
            'Alpha': alpha,
            'Delta_ModelValue_bp': delta_mv,
            'Delta_NonArb_bp': row['NonArb_Value_bp_mod'] - row['NonArb_Value_bp_orig'],
            'Delta_IQR_bp': delta_iqr,
        })

    return pd.DataFrame(summary_rows).sort_values(['Strategy', 'Alpha']).reset_index(drop=True)
