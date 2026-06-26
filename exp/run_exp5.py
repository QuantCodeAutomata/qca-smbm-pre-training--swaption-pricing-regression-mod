"""Exp-5: Cross-Parameter Robustness Validation."""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import ALPHA_LEVELS, STRATEGIES, OMEGA
from src.mc_simulation import load_paths
from src.deep_hedging import BermudanHedger
from src.evaluation import save_tables, run_consistency_checks
from src.robustness import (
    OMEGA_MOD, generate_modified_paths, evaluate_robustness,
    compute_robustness_summary, verify_atm_price_consistency,
    build_modified_correlation_matrices,
)


def run_exp5(
    save_dir: str = 'data',
    results_dir: str = 'results',
    trained_models=None,
    swaption_models=None,
    alpha_levels=None,
    verbose: bool = True,
) -> dict:
    """Run full exp_5: evaluate robustness under modified parameters."""
    os.makedirs(results_dir, exist_ok=True)

    if alpha_levels is None:
        alpha_levels = ALPHA_LEVELS

    if verbose:
        print("=" * 60)
        print("Exp-5: Cross-Parameter Robustness Validation")
        print("=" * 60)
        print(f"Original omega: {OMEGA:.4f}")
        print(f"Modified omega: {OMEGA_MOD:.5f} (halved)")
        print("S-X correlations: sign-flipped")
        print("X_0: unchanged (= 0)")

    # Step 1: Verify ATM price consistency under parameter modification
    if verbose:
        print("\n--- ATM Price Consistency Check (independent of omega) ---")
    atm_check = verify_atm_price_consistency(omega_new=OMEGA_MOD, n_verify=500)

    # Step 2: Build modified correlation matrices
    mod_corr_matrices = build_modified_correlation_matrices()
    if verbose:
        print("\n--- Modified Correlation Matrices ---")
        for p, C in enumerate(mod_corr_matrices):
            eigvals = np.linalg.eigvalsh(C)
            n = C.shape[0] // 2
            print(f"  Period {p} ({n} active): min_eigval={eigvals.min():.6f}, "
                  f"within-factor rho={C[0, n]:.3f} (should be +0.259 for flipped)")

    # Load swaption models
    if swaption_models is None:
        try:
            from src.swaption_pricing import load_swaption_models
            swaption_models = load_swaption_models(save_dir)
        except FileNotFoundError:
            swaption_models = None

    # Step 3: Generate modified paths
    from src.params import N_PATHS_TRAIN
    mod_paths = generate_modified_paths(
        swaption_models=swaption_models,
        save_dir=save_dir,
        n_paths=N_PATHS_TRAIN,
        verbose=verbose,
    )

    if verbose:
        print(f"\nModified paths shape: {mod_paths['S_rates'].shape}")
        # Initial prices check
        O002_t0 = mod_paths['swaption_K002'][:, 0, :]
        print("Modified swaption prices at t=0:")
        for i in range(5):
            print(f"  O^{i+1}(t=0) = {np.mean(O002_t0[:, i])*1e4:.2f} bp")

    # Load trained models
    if trained_models is None:
        trained_models = {}
        for strategy in STRATEGIES:
            for alpha in alpha_levels:
                model_path = os.path.join(save_dir, f'model_{strategy}_alpha{alpha:.1f}.pt')
                if os.path.exists(model_path):
                    hedger = BermudanHedger(strategy=strategy, alpha=alpha)
                    hedger.load(model_path)
                    trained_models[(strategy, alpha)] = hedger

    if not trained_models:
        if verbose:
            print("Warning: No trained models found. Run exp_3 first.")
        return {}

    # Step 4: Evaluate on modified paths
    table3, table4 = evaluate_robustness(
        trained_models=trained_models,
        paths_modified=mod_paths,
        alpha_levels=alpha_levels,
        basis_points_factor=10000.0,
    )

    # Save tables
    save_tables(table3, table4, save_dir=results_dir, prefix='mod_')

    if verbose:
        print("\n=== Table 3: Model Values (Modified Parameters) ===")
        print(table3.to_string(index=False))
        print("\n=== Table 4: P&L Distribution (Modified Parameters) ===")
        print(table4.to_string(index=False))

    # Run consistency checks on modified scenario
    checks_mod = run_consistency_checks(table3, table4)
    if verbose:
        print("\n--- Consistency Checks (Modified) ---")
        for name, passed in checks_mod.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name}: {status}")

    # Step 5: Compare original vs modified (requires table1/table2 from exp_4)
    t1_path = os.path.join(results_dir, 'table1.csv')
    t2_path = os.path.join(results_dir, 'table2.csv')
    summary = None
    if os.path.exists(t1_path) and os.path.exists(t2_path):
        table1 = pd.read_csv(t1_path)
        table2 = pd.read_csv(t2_path)
        summary = compute_robustness_summary(table1, table3, table2, table4)
        summary.to_csv(os.path.join(results_dir, 'robustness_summary.csv'), index=False)
        if verbose:
            print("\n=== Robustness Summary (change from original to modified) ===")
            print(summary.to_string(index=False))

    # Visualization
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        strategies_present = table3['Strategy'].unique()

        for strategy in strategies_present:
            mask3 = table3['Strategy'] == strategy
            ax_data3 = table3[mask3]
            axes[0].plot(ax_data3['Alpha'], ax_data3['Model_Value_bp'],
                        'o-', label=f'{strategy} (mod)', linewidth=2, linestyle='--')

        axes[0].set_xlabel('CVaR Alpha')
        axes[0].set_ylabel('Model Value (bp)')
        axes[0].set_title('Model Value: Modified Parameters')
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        for strategy in strategies_present:
            mask4 = table4['Strategy'] == strategy
            ax_data4 = table4[mask4]
            axes[1].plot(ax_data4['Alpha'], ax_data4['IQR_bp'],
                        'o-', label=f'{strategy} (mod)', linewidth=2, linestyle='--')

        axes[1].set_xlabel('CVaR Alpha')
        axes[1].set_ylabel('IQR (bp)')
        axes[1].set_title('IQR: Modified Parameters')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Cross-Parameter Robustness (omega halved, correlations flipped)', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'exp5_robustness.png'), dpi=100, bbox_inches='tight')
        plt.close()
        if verbose:
            print(f"\nPlot saved: {results_dir}/exp5_robustness.png")

    except Exception as e:
        if verbose:
            print(f"Warning: Plot failed: {e}")

    return {
        'table3': table3,
        'table4': table4,
        'summary': summary,
        'consistency_checks': checks_mod,
        'atm_consistency': atm_check,
    }


if __name__ == '__main__':
    results = run_exp5()
    if results:
        print("\n=== Exp-5 Complete ===")
