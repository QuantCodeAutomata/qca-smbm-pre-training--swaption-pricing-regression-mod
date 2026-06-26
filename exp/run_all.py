"""Master script: Run all experiments sequentially and save results to RESULTS.md."""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import N_GRID, N_INNER, N_PATHS_TOTAL, N_EPOCHS, ALPHA_LEVELS, STRATEGIES


def main():
    t_start = time.time()
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 70)
    print("SMBM Deep Hedging: Full Experiment Suite")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  N_grid={N_GRID} per dim ({N_GRID**3} grid pts per model)")
    print(f"  N_inner={N_INNER} (inner MC paths)")
    print(f"  N_paths={N_PATHS_TOTAL} total ({N_PATHS_TOTAL//2} train + {N_PATHS_TOTAL//2} test)")
    print(f"  N_epochs={N_EPOCHS} per component")
    print(f"  Strategies: {STRATEGIES}")
    print(f"  Alpha levels: {ALPHA_LEVELS}")
    print()

    # -------------------------------------------------------
    # Exp-1: Swaption Pricing Models
    # -------------------------------------------------------
    from exp.run_exp1 import run_exp1
    print("\n" + "=" * 70)
    print("Running Exp-1: Swaption Pricing Regression Models")
    print("=" * 70)
    t1 = time.time()
    exp1_results = run_exp1(save_dir='data', results_dir='results')
    models = exp1_results['models']
    t1_end = time.time()
    print(f"Exp-1 completed in {t1_end - t1:.1f}s")

    # -------------------------------------------------------
    # Exp-2: Monte Carlo Path Generation
    # -------------------------------------------------------
    from exp.run_exp2 import run_exp2
    print("\n" + "=" * 70)
    print("Running Exp-2: Monte Carlo Path Generation")
    print("=" * 70)
    t2 = time.time()
    exp2_results = run_exp2(save_dir='data', results_dir='results', swaption_models=models)
    train_paths = exp2_results['train_paths']
    test_paths = exp2_results['test_paths']
    max_swaption_bp = exp2_results['max_swaption_t0_bp']
    t2_end = time.time()
    print(f"Exp-2 completed in {t2_end - t2:.1f}s")

    # -------------------------------------------------------
    # Exp-3: Deep Hedging Training
    # -------------------------------------------------------
    from exp.run_exp3 import run_exp3
    print("\n" + "=" * 70)
    print("Running Exp-3: Deep Hedging Training")
    print("=" * 70)
    t3 = time.time()
    exp3_results = run_exp3(
        save_dir='data',
        results_dir='results',
        train_paths=train_paths,
        n_epochs=N_EPOCHS,
    )
    trained_models = exp3_results['trained_models']
    t3_end = time.time()
    print(f"Exp-3 completed in {t3_end - t3:.1f}s")

    # -------------------------------------------------------
    # Exp-4: Evaluation on Original Scenario
    # -------------------------------------------------------
    from exp.run_exp4 import run_exp4
    print("\n" + "=" * 70)
    print("Running Exp-4: P&L Distribution Analysis")
    print("=" * 70)
    t4 = time.time()
    exp4_results = run_exp4(
        save_dir='data',
        results_dir='results',
        trained_models=trained_models,
        test_paths=test_paths,
    )
    t4_end = time.time()
    print(f"Exp-4 completed in {t4_end - t4:.1f}s")

    # -------------------------------------------------------
    # Exp-5: Cross-Parameter Robustness
    # -------------------------------------------------------
    from exp.run_exp5 import run_exp5
    print("\n" + "=" * 70)
    print("Running Exp-5: Cross-Parameter Robustness")
    print("=" * 70)
    t5 = time.time()
    exp5_results = run_exp5(
        save_dir='data',
        results_dir='results',
        trained_models=trained_models,
        swaption_models=models,
    )
    t5_end = time.time()
    print(f"Exp-5 completed in {t5_end - t5:.1f}s")

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # -------------------------------------------------------
    # Write RESULTS.md
    # -------------------------------------------------------
    _write_results_md(
        exp1_results, exp2_results, exp4_results, exp5_results,
        total_time, max_swaption_bp,
    )

    print("\n" + "=" * 70)
    print("All experiments completed. Results saved in results/RESULTS.md")
    print("=" * 70)


def _write_results_md(
    exp1_results, exp2_results, exp4_results, exp5_results,
    total_time, max_swaption_bp,
):
    """Write RESULTS.md summary file."""
    bs_results = exp1_results.get('bs_consistency', {})
    val_results = exp1_results.get('validation', {})
    train_val = exp2_results.get('train_validation', {})
    test_val = exp2_results.get('test_validation', {})
    table1 = exp4_results.get('table1') if exp4_results else None
    table2 = exp4_results.get('table2') if exp4_results else None
    table3 = exp5_results.get('table3') if exp5_results else None
    table4 = exp5_results.get('table4') if exp5_results else None
    checks = exp4_results.get('consistency_checks', {}) if exp4_results else {}

    lines = [
        "# SMBM Deep Hedging: Experiment Results",
        "",
        "## Overview",
        "",
        "This report summarizes all 5 experiments for the Swap Market Bergomi Model (SMBM) "
        "deep hedging framework, including swaption pricing regression, Monte Carlo simulation, "
        "deep hedging training, evaluation, and cross-parameter robustness validation.",
        "",
        f"**Total runtime**: {total_time:.1f}s ({total_time/60:.1f} min)",
        "",
        "---",
        "",
        "## Exp-1: SMBM Swaption Pricing Regression Models",
        "",
        "Pre-trained 15 regression models (5 components × 3 strikes) using inner Monte Carlo",
        "simulation under the annuity measure A^{i,e} with scipy RegularGridInterpolator.",
        "",
        "### ATM Black-Scholes Consistency Check",
        "",
        "| Component i | T_i (y) | xi_0 | BS Price (bp) | Model Price (bp) | Rel Error |",
        "|------------|---------|------|---------------|-----------------|-----------|",
    ]

    for i in range(5):
        if i in bs_results:
            res = bs_results[i]
            lines.append(
                f"| {i+1} | {res['T_i']:.0f} | {res['xi0_i']:.3e} | "
                f"{res['bs_price_bp']:.3f} | {res['model_price_bp']:.3f} | "
                f"{res['rel_error']:.4%} |"
            )

    lines.extend([
        "",
        "### Validation (Mean Relative Error on Held-out Points)",
        "",
        "| Component i | K | Mean Rel Error | Max Rel Error |",
        "|------------|---|----------------|---------------|",
    ])

    for (i, k), val in val_results.items():
        lines.append(
            f"| {i+1} | {k:.3f} | {val['mean_rel_error']:.4%} | {val['max_rel_error']:.4%} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Exp-2: Monte Carlo Path Generation",
        "",
        f"Generated {N_PATHS_TOTAL} paths (half train, half test) under terminal measure P^e.",
        "",
        "### Path Validation",
        "",
        f"- Annuity positivity (min value): {train_val.get('annuity_min', 'N/A'):.6f}",
        f"- Martingale test (max |E[dU]|): {train_val.get('mean_abs_delta_swap_max', 'N/A'):.6f}",
        f"- Max swaption value at t=0: **{max_swaption_bp:.2f} bp**",
        f"  (expected: ~397 bp for full 4096-path simulation)",
        f"- Swaption non-negativity: {train_val.get('swaption_nonneg_pass', 'N/A')}",
        "",
        "---",
        "",
        "## Exp-3: Deep Hedging Training",
        "",
        f"Trained 16 configurations: {len(STRATEGIES)} strategies × {len(ALPHA_LEVELS)} alpha levels.",
        "",
        "**Strategies:**",
        "- **S_OS**: Option Spread Hedge (single spread asset G^i = O_bar^{i,e,K} - H^{i+1})",
        "- **S_I**: Swap-only hedge",
        "- **S_IS**: Swap + ATM swaption hedge",
        "- **S_ISM**: Swap + multi-strike swaption hedge",
        "- **S_Max**: Analytical benchmark (always holds max-value swaption)",
        "",
        "**Training configuration:**",
        f"- Network: 4 hidden layers × 32 units, BatchNorm + Dropout(0.5) + ReLU",
        f"- Loss: CVaR (Rockafellar-Uryasev) with learnable threshold z",
        f"- Optimizer: Adam (lr=1e-3), ReduceLROnPlateau",
        f"- Epochs per component: {N_EPOCHS}",
        f"- Gradient clipping: max_norm=1.0",
        "",
        "---",
        "",
        "## Exp-4: P&L Distribution Analysis (Original Scenario)",
        "",
        f"Evaluated all 20 configurations on {N_PATHS_TOTAL//2} held-out test paths.",
        "",
    ])

    if table1 is not None:
        lines.extend([
            "### Table 1: Model Values and Switch Values",
            "",
            table1.to_markdown(index=False, floatfmt='.2f'),
            "",
            "### Table 2: P&L Drawdown Distribution",
            "",
            table2.to_markdown(index=False, floatfmt='.2f'),
            "",
        ])

    lines.extend([
        "### Consistency Checks",
        "",
    ])
    for name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"- **{name}**: {status}")

    lines.extend([
        "",
        "**Key findings (expected from paper):**",
        f"- Max swaption value at t=0: {max_swaption_bp:.2f} bp (paper: ~397 bp)",
        "- NonArb Values ~455-457 bp across all configurations",
        "- Swaption hedging dramatically reduces IQR vs swap-only",
        "- S^Max IQR constant (~73 bp) across all alpha levels",
        "- S^OS limits downside to option spread switch value",
        "",
        "---",
        "",
        "## Exp-5: Cross-Parameter Robustness",
        "",
        "Evaluated all trained models on paths generated with modified parameters:",
        "- vol-of-vol halved: omega_mod = 0.42015",
        "- Rate-vol correlations sign-flipped",
        "- X_0 unchanged (ATM prices independent of omega)",
        "",
    ])

    if table3 is not None:
        lines.extend([
            "### Table 3: Model Values (Modified Parameters)",
            "",
            table3.to_markdown(index=False, floatfmt='.2f'),
            "",
            "### Table 4: P&L Distribution (Modified Parameters)",
            "",
            table4.to_markdown(index=False, floatfmt='.2f'),
            "",
        ])

    lines.extend([
        "**Key findings (expected from paper):**",
        "- Swaption-based strategies (S^OS, S^IS, S^ISM) remain more robust than S^I",
        "- S^Max IQR ~70.5 bp (lower than 73 bp due to halved vol-vol)",
        "- CVaR99 for S^IS and S^ISM improves due to lower vol-vol",
        "- S^OS Model Values close to original values",
        "",
        "---",
        "",
        "## Figures",
        "",
        "- `results/exp1_swaption_prices.png`: Swaption prices as function of swap rate",
        "- `results/exp1_bs_comparison.png`: ATM Black-Scholes vs regression model",
        "- `results/exp2_mc_paths.png`: Monte Carlo paths visualization",
        "- `results/exp3_training_results.png`: Training results",
        "- `results/exp4_model_values.png`: Model values by strategy and alpha",
        "- `results/exp4_tail_risk.png`: Tail risk (CVaR99) by strategy",
        "- `results/exp5_robustness.png`: Robustness comparison",
        "",
        "---",
        "",
        "## Implementation Notes",
        "",
        "- **Swaption pricing**: scipy RegularGridInterpolator (linear, extrapolating)",
        "- **MC simulation**: Euler-Maruyama with 5-period Cholesky correlation structure",
        "- **Deep hedging**: PyTorch feedforward networks with CVaR (Rockafellar-Uryasev)",
        "- **Bootstrap**: Recursive formula for P^u/P^e and A_bar^{u,e}",
        "- **Drift correction**: Numerical Jacobian of log-annuity factors",
        "- **Correlation matrices**: Block structure with within-factor rho = -0.259",
        "",
        f"*Note: Reduced parameters used for computational feasibility:*",
        f"*N_grid={N_GRID}/dim (vs 30), N_inner={N_INNER} (vs 100k), N_paths={N_PATHS_TOTAL} (vs 8192), "
        f"N_epochs={N_EPOCHS} (vs 500). Full paper parameters require ~24 hrs runtime.*",
    ])

    with open('results/RESULTS.md', 'w') as f:
        f.write('\n'.join(lines))

    print("RESULTS.md written.")


if __name__ == '__main__':
    main()
