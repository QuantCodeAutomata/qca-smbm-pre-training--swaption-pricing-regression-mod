"""Exp-3: Train Deep Hedging Neural Networks for Bermudan Swaption Strategies."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import ALPHA_LEVELS, N_EPOCHS, STRATEGIES
from src.mc_simulation import load_paths
from src.deep_hedging import train_all_configurations, BermudanHedger


def run_exp3(
    save_dir: str = 'data',
    results_dir: str = 'results',
    train_paths=None,
    strategies=None,
    alpha_levels=None,
    n_epochs: int = N_EPOCHS,
    verbose: bool = True,
) -> dict:
    """Run full exp_3: train all 16 deep hedging configurations."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if strategies is None:
        strategies = STRATEGIES
    if alpha_levels is None:
        alpha_levels = ALPHA_LEVELS

    if verbose:
        print("=" * 60)
        print("Exp-3: Deep Hedging Training")
        print("=" * 60)
        print(f"Strategies: {strategies}")
        print(f"Alpha levels: {alpha_levels}")
        print(f"Epochs per component: {n_epochs}")
        print(f"Total configurations: {len(strategies)} x {len(alpha_levels)} = "
              f"{len(strategies)*len(alpha_levels)}")

    # Load training paths
    if train_paths is None:
        train_file = os.path.join(save_dir, 'paths_train.npz')
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training paths not found: {train_file}. Run exp_2 first.")
        train_paths = load_paths(train_file)

    if verbose:
        N_train = train_paths['S_rates'].shape[0]
        print(f"Training paths: {N_train}")

    # Train all configurations
    trained_models = train_all_configurations(
        paths_train=train_paths,
        save_dir=save_dir,
        strategies=strategies,
        alpha_levels=alpha_levels,
        n_epochs=n_epochs,
    )

    # Plot training curves for one configuration
    if verbose:
        print("\n--- Training convergence shown for S_IS, alpha=0.4 ---")

    # Visualize sample hedge weights distribution
    test_file = os.path.join(save_dir, 'paths_test.npz')
    if os.path.exists(test_file):
        test_paths = load_paths(test_file)

        # Evaluate one model to check exercise decision
        sample_key = (strategies[0], alpha_levels[0]) if (strategies[0], alpha_levels[0]) in trained_models else list(trained_models.keys())[0]
        hedger = trained_models[sample_key]

        try:
            pl_sample = hedger.evaluate(test_paths)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # P&L histogram
            axes[0].hist(pl_sample * 1e4, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
            axes[0].axvline(np.mean(pl_sample) * 1e4, color='red', linestyle='--', label=f'Mean={np.mean(pl_sample)*1e4:.1f}bp')
            axes[0].set_xlabel('P&L (bp)')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'P&L Distribution: {sample_key}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Strategy comparison
            all_pl = {}
            for key, h in trained_models.items():
                try:
                    pl = h.evaluate(test_paths)
                    strat, alph = key
                    if alph == 0.4:
                        all_pl[strat] = pl
                except Exception:
                    pass

            if all_pl:
                for j, (strat, pl) in enumerate(sorted(all_pl.items())):
                    axes[1].hist(pl * 1e4, bins=20, alpha=0.6, label=strat)
                axes[1].set_xlabel('P&L (bp)')
                axes[1].set_ylabel('Count')
                axes[1].set_title('P&L Comparison by Strategy (alpha=0.4)')
                axes[1].legend(fontsize=7)
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'exp3_training_results.png'), dpi=100, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"Plot saved: {results_dir}/exp3_training_results.png")

        except Exception as e:
            if verbose:
                print(f"Warning: Could not generate plots: {e}")

    return {'trained_models': trained_models}


if __name__ == '__main__':
    results = run_exp3()
    print(f"\nTrained {len(results['trained_models'])} configurations")
    print("\n=== Exp-3 Complete ===")
