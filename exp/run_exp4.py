"""Exp-4: Evaluate Deep Hedging Models on Original Scenario Test Set."""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import ALPHA_LEVELS, STRATEGIES, EXERCISE_IDX
from src.mc_simulation import load_paths
from src.deep_hedging import BermudanHedger, smax_strategy
from src.evaluation import (
    evaluate_all_configurations, run_consistency_checks, save_tables,
    compute_cvar,
)


def run_exp4(
    save_dir: str = 'data',
    results_dir: str = 'results',
    trained_models=None,
    test_paths=None,
    alpha_levels=None,
    verbose: bool = True,
) -> dict:
    """Run full exp_4: evaluate all configurations on test paths."""
    os.makedirs(results_dir, exist_ok=True)

    if alpha_levels is None:
        alpha_levels = ALPHA_LEVELS

    if verbose:
        print("=" * 60)
        print("Exp-4: Deep Hedging Evaluation (Original Scenario)")
        print("=" * 60)

    # Load test paths
    if test_paths is None:
        test_file = os.path.join(save_dir, 'paths_test.npz')
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test paths not found: {test_file}. Run exp_2 first.")
        test_paths = load_paths(test_file)

    if verbose:
        print(f"Test paths: {test_paths['S_rates'].shape[0]}")

    # Load trained models if not provided
    if trained_models is None:
        trained_models = {}
        for strategy in STRATEGIES:
            for alpha in alpha_levels:
                model_path = os.path.join(save_dir, f'model_{strategy}_alpha{alpha:.1f}.pt')
                if os.path.exists(model_path):
                    hedger = BermudanHedger(strategy=strategy, alpha=alpha)
                    hedger.load(model_path)
                    trained_models[(strategy, alpha)] = hedger
                    if verbose:
                        print(f"  Loaded: {strategy}, alpha={alpha}")

    if not trained_models:
        if verbose:
            print("Warning: No trained models found. Run exp_3 first.")
        return {}

    # Evaluate all configurations
    table1, table2 = evaluate_all_configurations(
        trained_models=trained_models,
        paths_test=test_paths,
        alpha_levels=alpha_levels,
        basis_points_factor=10000.0,
    )

    # Save tables
    save_tables(table1, table2, save_dir=results_dir, prefix='')

    if verbose:
        print("\n=== Table 1: Model Values and Switch Values ===")
        print(table1.to_string(index=False))
        print("\n=== Table 2: P&L Drawdown Distribution ===")
        print(table2.to_string(index=False))

    # Run consistency checks
    checks = run_consistency_checks(table1, table2)
    if verbose:
        print("\n--- Consistency Checks ---")
        for name, passed in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name}: {status}")

    # Visualizations
    _plot_pl_distributions(table1, table2, test_paths, trained_models, alpha_levels, results_dir)

    return {
        'table1': table1,
        'table2': table2,
        'consistency_checks': checks,
    }


def _plot_pl_distributions(table1, table2, test_paths, trained_models, alpha_levels, results_dir):
    """Generate P&L distribution plots."""
    try:
        # Plot 1: Model Values by strategy
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        strategies_present = table1['Strategy'].unique()

        for strategy in strategies_present:
            mask = table1['Strategy'] == strategy
            ax_data = table1[mask]
            axes[0].plot(ax_data['Alpha'], ax_data['Model_Value_bp'],
                        'o-', label=strategy, linewidth=2)

        axes[0].set_xlabel('CVaR Alpha')
        axes[0].set_ylabel('Model Value (bp)')
        axes[0].set_title('Model Value by Strategy and Alpha')
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: IQR by strategy
        for strategy in strategies_present:
            mask = table2['Strategy'] == strategy
            ax_data = table2[mask]
            axes[1].plot(ax_data['Alpha'], ax_data['IQR_bp'],
                        'o-', label=strategy, linewidth=2)

        axes[1].set_xlabel('CVaR Alpha')
        axes[1].set_ylabel('IQR (bp)')
        axes[1].set_title('P&L IQR by Strategy and Alpha')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Deep Hedging Evaluation: Original Scenario', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'exp4_model_values.png'), dpi=100, bbox_inches='tight')
        plt.close()

        # Plot 2: CVaR tail metrics
        fig, ax = plt.subplots(figsize=(10, 5))
        for strategy in strategies_present:
            mask = table2['Strategy'] == strategy
            ax_data = table2[mask]
            ax.plot(ax_data['Alpha'], ax_data['CVaR99_bp'],
                   'o-', label=f'{strategy} CVaR99', linewidth=2)

        ax.set_xlabel('CVaR Alpha')
        ax.set_ylabel('CVaR99 (bp)')
        ax.set_title('Tail Risk (CVaR99) by Strategy and Alpha')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'exp4_tail_risk.png'), dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Plots saved in {results_dir}/")

    except Exception as e:
        print(f"Warning: Plot generation failed: {e}")


if __name__ == '__main__':
    results = run_exp4()
    if results:
        print("\n=== Exp-4 Complete ===")
