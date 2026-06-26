"""Exp-1: Pre-train SMBM Swaption Pricing Regression Models.

Trains 15 regression models (5 components x 3 strikes) and validates them.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import XI0, T_EXERCISE, STRIKES, N_GRID, N_INNER
from src.swaption_pricing import (
    train_swaption_models, validate_swaption_model,
    validate_bs_consistency, price_swaption,
)
from src.black_scholes import bs_receiver_swaption


def run_exp1(
    save_dir: str = 'data',
    results_dir: str = 'results',
    N_grid: int = N_GRID,
    N_inner: int = N_INNER,
    verbose: bool = True,
) -> dict:
    """Run full exp_1: train and validate swaption pricing models."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Exp-1: SMBM Swaption Pricing Regression Models")
        print("=" * 60)
        print(f"Grid size: {N_grid}^3 = {N_grid**3} points per (i,k)")
        print(f"Inner MC paths: {N_inner}")
        print(f"Total (i,k) pairs: 5 x 3 = 15")
        print()

    # Step 1-3: Train models
    models = train_swaption_models(
        save_dir=save_dir,
        N_grid=N_grid,
        N_inner=N_inner,
        verbose=verbose if verbose else False,
    )

    # Step 4: Validate models
    if verbose:
        print("\n--- Validation Results ---")

    val_results = {}
    for i in range(5):
        k = 0.02  # ATM only for main validation
        model_data = models[(i, k)]
        val = validate_swaption_model(model_data, N_inner_val=N_inner)
        val_results[(i, k)] = val

        if verbose:
            print(f"  i={i+1}, k={k}: mean_rel_err={val['mean_rel_error']:.4%}, "
                  f"max_rel_err={val['max_rel_error']:.4%}")

    # Step 6: ATM Black-Scholes consistency check
    if verbose:
        print("\n--- ATM Black-Scholes Consistency Check ---")

    bs_results = validate_bs_consistency(models)
    for i, res in bs_results.items():
        if verbose:
            print(f"  i={i+1}: BS={res['bs_price_bp']:.3f}bp, "
                  f"Model={res['model_price_bp']:.3f}bp, "
                  f"RelErr={res['rel_error']:.4%}")

    # Step 5: Boundary check (tau -> 0)
    if verbose:
        print("\n--- Boundary Condition Check (tau -> Delta_t) ---")

    boundary_results = {}
    for i in range(5):
        k = 0.02
        model_data = models[(i, k)]
        S_vals = np.linspace(0.01, 0.04, 5)
        tau_small = 1.0 / 32.0
        for S_val in [0.01, 0.025, 0.03]:
            v_model = price_swaption(model_data, S_val, 0.0, tau_small)
            v_intrinsic = max(k - S_val, 0.0)
            boundary_results[(i, S_val)] = {
                'model': v_model,
                'intrinsic': v_intrinsic,
                'diff': abs(v_model - v_intrinsic),
            }
            if verbose:
                print(f"  i={i+1}, S={S_val:.3f}: model={v_model*1e4:.2f}bp, "
                      f"intrinsic={v_intrinsic*1e4:.2f}bp")
        break  # show for i=1 only

    # Plot: swaption prices as function of S for different tau
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i in range(5):
        k = 0.02
        model_data = models[(i, k)]
        T_i = T_EXERCISE[i]
        S_vals = np.linspace(0.005, 0.04, 50)

        for tau in [T_i * 0.25, T_i * 0.5, T_i]:
            prices = [price_swaption(model_data, S, 0.0, tau) * 1e4 for S in S_vals]
            axes[i].plot(S_vals * 1e4, prices, label=f'tau={tau:.1f}y')

        axes[i].set_title(f'i={i+1}, T={T_i:.0f}y, K=2%')
        axes[i].set_xlabel('Swap Rate S (bp)')
        axes[i].set_ylabel('Swaption Price (bp)')
        axes[i].legend(fontsize=7)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('SMBM Swaption Prices: Model Regression', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp1_swaption_prices.png'), dpi=100, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"\nPlot saved: {results_dir}/exp1_swaption_prices.png")

    # BS comparison plot
    fig, ax = plt.subplots(figsize=(8, 5))
    components = range(1, 6)
    bs_prices = [bs_results[i]['bs_price_bp'] for i in range(5)]
    model_prices = [bs_results[i]['model_price_bp'] for i in range(5)]

    ax.plot(list(components), bs_prices, 'bo-', label='Black-Scholes', linewidth=2)
    ax.plot(list(components), model_prices, 'r^--', label='Regression Model', linewidth=2)
    ax.set_xlabel('Component i')
    ax.set_ylabel('ATM Swaption Price (bp)')
    ax.set_title('ATM Swaption: Black-Scholes vs Regression Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp1_bs_comparison.png'), dpi=100, bbox_inches='tight')
    plt.close()

    results = {
        'models': models,
        'validation': val_results,
        'bs_consistency': bs_results,
        'boundary': boundary_results,
    }

    return results


if __name__ == '__main__':
    results = run_exp1()
    print("\n=== Exp-1 Complete ===")
