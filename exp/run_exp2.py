"""Exp-2: Generate SMBM Monte Carlo Paths under Terminal Measure P^e."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import N_PATHS_TOTAL, SEED, SEED_TEST, XI0, T_EXERCISE, EXERCISE_IDX
from src.swaption_pricing import load_swaption_models
from src.mc_simulation import generate_all_paths, validate_paths


def run_exp2(
    save_dir: str = 'data',
    results_dir: str = 'results',
    swaption_models=None,
    n_total: int = N_PATHS_TOTAL,
    verbose: bool = True,
) -> dict:
    """Run full exp_2: generate and validate MC paths."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Exp-2: SMBM Monte Carlo Path Generation")
        print("=" * 60)
        print(f"Total paths: {n_total} ({n_total//2} train + {n_total//2} test)")

    # Load pre-trained swaption models
    if swaption_models is None:
        try:
            swaption_models = load_swaption_models(save_dir)
            if verbose:
                print("Loaded pre-trained swaption models.")
        except FileNotFoundError:
            if verbose:
                print("No pre-trained models found. Using Black-Scholes fallback.")
            swaption_models = None

    # Generate paths
    train_paths, test_paths = generate_all_paths(
        swaption_models=swaption_models,
        save_dir=save_dir,
        n_total=n_total,
        seed_train=SEED,
        seed_test=SEED_TEST,
        verbose=verbose,
    )

    # Validate paths
    if verbose:
        print("\n--- Validating Training Paths ---")
    train_val = validate_paths(train_paths, verbose=verbose)

    if verbose:
        print("\n--- Validating Test Paths ---")
    test_val = validate_paths(test_paths, verbose=verbose)

    # Compute initial ATM prices
    O002_t0 = test_paths['swaption_K002'][:, 0, :]  # (N, 5)
    bs_atm_bp = [xi * T * 1e4 for xi, T in zip(XI0, T_EXERCISE)]

    if verbose:
        print("\n--- Initial ATM Swaption Prices at t=0 ---")
        for i in range(5):
            mean_price = np.mean(O002_t0[:, i]) * 1e4
            print(f"  O^{i+1},e,K=0.02(t=0) = {mean_price:.2f} bp")

        max_swaption = np.max(np.mean(O002_t0, axis=0)) * 1e4
        print(f"  max_j O^j,e,K=0.02(t=0) = {max_swaption:.2f} bp")

    # Plot swap rate paths
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    n_show = min(50, test_paths['S_rates'].shape[0])
    t_grid = np.arange(289) / 32.0

    for i in range(5):
        ax = axes[i]
        # Only show up to exercise date
        end_step = EXERCISE_IDX[i]
        for n in range(n_show):
            ax.plot(t_grid[:end_step], test_paths['S_rates'][n, :end_step, i] * 1e4,
                    alpha=0.1, color='blue', linewidth=0.5)
        ax.axhline(200, color='red', linestyle='--', label='K=2%')
        ax.set_title(f'S^{i+1},e paths (T={T_EXERCISE[i]:.0f}y)')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Swap Rate (bp)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Plot swaption prices
    ax = axes[5]
    for i in range(5):
        end_step = EXERCISE_IDX[i]
        mean_opt = np.mean(test_paths['swaption_K002'][:, :end_step, i], axis=0) * 1e4
        ax.plot(t_grid[:end_step], mean_opt, label=f'O^{i+1}')
    ax.set_title('Mean ATM Swaption Prices')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Price (bp)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('SMBM Monte Carlo Paths under P^e', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp2_mc_paths.png'), dpi=100, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nPlot saved: {results_dir}/exp2_mc_paths.png")

    results = {
        'train_paths': train_paths,
        'test_paths': test_paths,
        'train_validation': train_val,
        'test_validation': test_val,
        'max_swaption_t0_bp': np.max(np.mean(O002_t0, axis=0)) * 1e4,
    }

    return results


if __name__ == '__main__':
    results = run_exp2()
    print(f"\nMax swaption t=0: {results['max_swaption_t0_bp']:.2f} bp")
    print("\n=== Exp-2 Complete ===")
