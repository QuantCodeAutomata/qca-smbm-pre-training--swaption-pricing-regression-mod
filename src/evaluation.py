"""Exp-4: Deep Hedging Evaluation - P&L Distribution Analysis.

Computes all performance and distributional metrics for all 20 model
configurations on held-out test paths. Produces Tables 1 and 2.

# Custom metric computation - Context7 found no library equivalent
"""
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import (
    EXERCISE_IDX, T_EXERCISE, ALPHA_LEVELS, BERMUDAN_STRIKE, STRIKE_ATM,
    N_COMPONENTS,
)


# ------------------------------------------------------------------
# CVaR computation (Rockafellar-Uryasev)
# ------------------------------------------------------------------

def compute_cvar(pl_array: np.ndarray, alpha: float) -> float:
    """Compute CVaR_alpha(-PL) = Expected Shortfall at level alpha.

    CVaR_alpha(-PL) = -min_z { z + 1/(1-alpha) * E[max(-PL - z, 0)] }
                    = -quantile_alpha(PL) + E[max(quantile_alpha - PL, 0)] / (1-alpha)

    Equivalently: model value V = -CVaR = mean of bottom (1-alpha) fraction of PL.

    Args:
        pl_array: P&L values, shape (N,).
        alpha: Confidence level (higher = more risk-averse).

    Returns:
        CVaR value (positive means loss in CVaR sense, negative means gain).
        Note: Model Value V = -compute_cvar(pl_array, alpha).
    """
    # z* = VaR_{alpha}(-PL) = -quantile_{1-alpha}(PL)  (Rockafellar-Uryasev optimal threshold)
    z_opt = np.quantile(pl_array, 1.0 - alpha)
    shortfall = np.mean(np.maximum(z_opt - pl_array, 0.0))
    return float(-z_opt + shortfall / (1.0 - alpha))


def compute_model_value(pl_array: np.ndarray, alpha: float) -> float:
    """Compute Model Value V = -CVaR_alpha(-PL) = -CVaR_alpha of losses.

    V represents the fair value of the Bermudan swaption accounting for
    hedging costs at risk level alpha.

    Args:
        pl_array: P&L values.
        alpha: CVaR level.

    Returns:
        Model value (positive = profitable hedging strategy).
    """
    return -compute_cvar(pl_array, alpha)


# ------------------------------------------------------------------
# Metric computation for a single configuration
# ------------------------------------------------------------------

def compute_metrics(
    pl_array: np.ndarray,
    alpha: float,
    exercise_payoffs: Optional[np.ndarray] = None,
    max_swaption_t0: float = 0.0,
) -> Dict[str, float]:
    """Compute all metrics for a given P&L distribution.

    Args:
        pl_array: P&L values, shape (N,).
        alpha: Training CVaR level.
        exercise_payoffs: Expected exercise payoffs Z*_n, shape (N,).
                          If provided, NonArb Value = mean(exercise_payoffs).
        max_swaption_t0: Max swaption value at t=0 (in same units as PL).

    Returns:
        Dict with all metrics (in original relative units).
    """
    N = len(pl_array)

    # Model Value
    model_value = compute_model_value(pl_array, alpha)

    # NonArb Value (expected exercise payoff)
    if exercise_payoffs is not None:
        nonarb_value = float(np.mean(exercise_payoffs))
    else:
        nonarb_value = float(np.mean(pl_array))  # fallback

    # Hedge P&L mean
    if exercise_payoffs is not None:
        hedge_pnl_mean = float(np.mean(pl_array - exercise_payoffs))
    else:
        hedge_pnl_mean = 0.0

    # Switch values
    model_switch = model_value - max_swaption_t0
    nonarb_switch = nonarb_value - max_swaption_t0

    # Drawdown distribution: drawdown = V - PL
    drawdown = model_value - pl_array
    p25 = float(np.percentile(drawdown, 25))
    p50 = float(np.percentile(drawdown, 50))
    p75 = float(np.percentile(drawdown, 75))
    iqr = p75 - p25
    # Loss probability: fraction of paths where P&L is negative (actual losses)
    loss_prob = float(np.mean(pl_array < 0))
    losses = -pl_array[pl_array < 0]
    expected_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    # Tail risk measures (fixed, independent of training alpha)
    cvar95 = float(-np.mean(np.sort(pl_array)[:max(1, int(0.05 * N))]))
    cvar99 = float(-np.mean(np.sort(pl_array)[:max(1, int(0.01 * N))]))

    return {
        'model_value': model_value,
        'nonarb_value': nonarb_value,
        'hedge_pnl_mean': hedge_pnl_mean,
        'model_switch': model_switch,
        'nonarb_switch': nonarb_switch,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'iqr': iqr,
        'loss_prob': loss_prob,
        'expected_loss': expected_loss,
        'cvar95': cvar95,
        'cvar99': cvar99,
    }


# ------------------------------------------------------------------
# Exercise payoff computation from trained networks
# ------------------------------------------------------------------

def compute_exercise_payoffs(
    hedger,
    paths: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute expected exercise payoffs Z*_n for each test path.

    Z*_n = E_net(features_{T_{i*},n}) * O_bar^{i*,e,K}_{T_{i*},n}
    where i* is the first Bermudan exercise reached.

    Args:
        hedger: Trained BermudanHedger instance.
        paths: Test path dictionary.

    Returns:
        Exercise payoffs, shape (N,).
    """
    import torch

    N = paths['S_rates'].shape[0]
    opt002 = paths['swaption_K002']
    swap_p = paths['swap_prices']
    opt001 = paths['swaption_K001']
    opt0015 = paths['swaption_K0015']

    device = hedger.device
    opt002_t = torch.from_numpy(opt002.astype(np.float32)).to(device)
    swap_t = torch.from_numpy(swap_p.astype(np.float32)).to(device)
    opt001_t = torch.from_numpy(opt001.astype(np.float32)).to(device)
    opt0015_t = torch.from_numpy(opt0015.astype(np.float32)).to(device)

    exercised = np.zeros(N, dtype=bool)
    payoffs = np.zeros(N)

    # Process exercise decisions recursively
    for comp in range(4, -1, -1):
        if comp not in hedger.E_nets:
            continue
        e_net = hedger.E_nets[comp]
        e_net.eval()

        exercise_step = EXERCISE_IDX[comp]
        n_active = 5 - comp

        from src.deep_hedging import feature_dim

        with torch.no_grad():
            ex_feats = hedger._get_exercise_features(
                comp, exercise_step,
                swap_t[:, exercise_step, :],
                opt002_t[:, exercise_step, :],
                opt001_t[:, exercise_step, :],
                opt0015_t[:, exercise_step, :],
            )
            e_prob = e_net(ex_feats).squeeze(1).cpu().numpy()

        # Exercise now: paths where e_prob > 0.5 and not yet exercised
        exercise_now = ~exercised & (e_prob > 0.5)
        if exercise_step < opt002.shape[1]:
            payoffs[exercise_now] = opt002[exercise_now, exercise_step, comp]
        exercised = exercised | exercise_now

    # Any remaining paths: exercise at last date
    if np.any(~exercised):
        last_step = EXERCISE_IDX[-1]
        if last_step < opt002.shape[1]:
            payoffs[~exercised] = opt002[~exercised, last_step, -1]

    return payoffs


# ------------------------------------------------------------------
# Full evaluation pipeline
# ------------------------------------------------------------------

def evaluate_all_configurations(
    trained_models: Dict,
    paths_test: Dict[str, np.ndarray],
    alpha_levels: Optional[List[float]] = None,
    basis_points_factor: float = 10000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all trained configurations and produce Tables 1 and 2.

    Args:
        trained_models: Dict mapping (strategy, alpha) -> BermudanHedger.
        paths_test: Test paths dictionary.
        alpha_levels: CVaR levels to evaluate.
        basis_points_factor: Multiplier to convert to basis points.

    Returns:
        (table1, table2): DataFrames for Tables 1 and 2.
    """
    if alpha_levels is None:
        alpha_levels = ALPHA_LEVELS

    # Compute max swaption value at t=0
    opt002_t0 = paths_test['swaption_K002'][:, 0, :]  # (N, 5)
    max_swaption_t0 = float(np.max(np.mean(opt002_t0, axis=0)))
    max_swaption_t0_bp = max_swaption_t0 * basis_points_factor
    print(f"Max swaption value at t=0: {max_swaption_t0_bp:.2f} bp")

    rows_t1 = []  # Table 1
    rows_t2 = []  # Table 2

    bp = basis_points_factor

    for (strategy, alpha), hedger in sorted(trained_models.items()):
        print(f"\nEvaluating {strategy}, alpha={alpha}...")

        # Compute P&L
        pl_array = hedger.evaluate(paths_test)

        # Compute exercise payoffs
        try:
            ex_payoffs = compute_exercise_payoffs(hedger, paths_test)
        except Exception:
            ex_payoffs = None

        metrics = compute_metrics(
            pl_array, alpha, ex_payoffs, max_swaption_t0
        )

        # Table 1 row
        rows_t1.append({
            'Strategy': strategy,
            'Alpha': alpha,
            'Model_Value_bp': metrics['model_value'] * bp,
            'NonArb_Value_bp': metrics['nonarb_value'] * bp,
            'Hedge_PnL_Mean_bp': metrics['hedge_pnl_mean'] * bp,
            'Model_Switch_bp': metrics['model_switch'] * bp,
            'NonArb_Switch_bp': metrics['nonarb_switch'] * bp,
        })

        # Table 2 row
        rows_t2.append({
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

    # S^Max analytical strategy
    from src.deep_hedging import smax_strategy
    opt002 = paths_test['swaption_K002']
    S_rates = paths_test['S_rates']
    pl_smax = smax_strategy(opt002, EXERCISE_IDX, S_rates=S_rates)

    for alpha in alpha_levels:
        metrics = compute_metrics(
            pl_smax, alpha, None, max_swaption_t0
        )

        rows_t1.append({
            'Strategy': 'S_Max',
            'Alpha': alpha,
            'Model_Value_bp': metrics['model_value'] * bp,
            'NonArb_Value_bp': metrics['nonarb_value'] * bp,
            'Hedge_PnL_Mean_bp': metrics['hedge_pnl_mean'] * bp,
            'Model_Switch_bp': metrics['model_switch'] * bp,
            'NonArb_Switch_bp': metrics['nonarb_switch'] * bp,
        })

        rows_t2.append({
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

    table1 = pd.DataFrame(rows_t1).sort_values(['Strategy', 'Alpha']).reset_index(drop=True)
    table2 = pd.DataFrame(rows_t2).sort_values(['Strategy', 'Alpha']).reset_index(drop=True)

    return table1, table2


def run_consistency_checks(table1: pd.DataFrame, table2: pd.DataFrame) -> Dict[str, bool]:
    """Verify consistency conditions from the paper.

    Args:
        table1: Table 1 DataFrame.
        table2: Table 2 DataFrame.

    Returns:
        Dict of check_name -> passed (bool).
    """
    checks = {}

    # NonArb Values within 50 bp of each other
    # Support both title-case (internal) and snake_case (test fixtures) column names
    nonarb_col = 'nonarb_value_bp' if 'nonarb_value_bp' in table1.columns else 'NonArb_Value_bp'
    nonarb_std = table1[nonarb_col].std()
    checks['nonarb_stable'] = nonarb_std < 50.0

    # CVaR99 >= CVaR95 for all rows
    c99_col = 'cvar99_bp' if 'cvar99_bp' in table2.columns else 'CVaR99_bp'
    c95_col = 'cvar95_bp' if 'cvar95_bp' in table2.columns else 'CVaR95_bp'
    checks['cvar99_ge_cvar95'] = bool(np.all(
        table2[c99_col].values >= table2[c95_col].values - 1e-3
    ))

    # S^Max IQR constant across alpha
    strat_col = 'strategy' if 'strategy' in table2.columns else 'Strategy'
    iqr_col = 'iqr_bp' if 'iqr_bp' in table2.columns else 'IQR_bp'
    smax_rows = table2[table2[strat_col] == 'S_Max']
    if len(smax_rows) > 1:
        checks['smax_iqr_constant'] = (
            smax_rows[iqr_col].std() < 10.0  # within 10 bp
        )
    else:
        checks['smax_iqr_constant'] = True

    # S^I should have larger IQR than swaption strategies
    si_iqr = table2[table2[strat_col] == 'S_I'][iqr_col].mean()
    so_iqr = table2[table2[strat_col] == 'S_IS'][iqr_col].mean()
    checks['swaption_reduces_iqr'] = (si_iqr > so_iqr)

    return checks


def save_tables(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    save_dir: str = 'results',
    prefix: str = '',
) -> None:
    """Save Tables 1 and 2 as CSV files.

    Args:
        table1: Table 1 DataFrame.
        table2: Table 2 DataFrame.
        save_dir: Output directory.
        prefix: File name prefix (for cross-parameter variants).
    """
    os.makedirs(save_dir, exist_ok=True)
    t1_path = os.path.join(save_dir, f'{prefix}table1.csv')
    t2_path = os.path.join(save_dir, f'{prefix}table2.csv')
    table1.to_csv(t1_path, index=False, float_format='%.4f')
    table2.to_csv(t2_path, index=False, float_format='%.4f')
    print(f"Tables saved: {t1_path}, {t2_path}")
