"""Shared pytest fixtures for the SMBM deep hedging test suite."""
import sys
import os
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def rng():
    """Seeded RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def tiny_swaption_model():
    """A small pre-trained swaption model for i=0, k=0.02 (fast, N_grid=4, N_inner=200)."""
    from src.swaption_pricing import compute_swaption_grid, fit_swaption_model
    from src.params import XI0, T_EXERCISE
    i = 0
    k = 0.02
    S_nodes, X_nodes, tau_nodes, V_grid = compute_swaption_grid(
        component_idx=i, strike=k, N_grid=4, N_inner=200
    )
    interp = fit_swaption_model(S_nodes, X_nodes, tau_nodes, V_grid)
    return {
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


@pytest.fixture(scope="session")
def tiny_paths():
    """Tiny path set (16 paths) for unit testing without full MC cost."""
    from src.mc_simulation import simulate_paths
    paths = simulate_paths(n_paths=16, swaption_models=None, seed=42, verbose=False)
    return paths
