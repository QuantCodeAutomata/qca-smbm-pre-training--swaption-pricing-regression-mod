"""Model parameters and constants for the SMBM deep hedging experiment."""
import numpy as np
from typing import List

# --- SMBM Model Parameters ---
OMEGA: float = 0.8403            # vol-of-vol
KAPPA: float = 0.0               # mean reversion
S0: float = 0.02                 # initial swap rates (all i)
DELTA_T: float = 1.0 / 32.0     # time step (years)

# Initial variance states xi_0^{i,e} for i=1,...,5
XI0: np.ndarray = np.array([1.132e-04, 1.228e-04, 1.287e-04, 1.344e-04, 1.451e-04])

# Exercise dates T_i = i+3 for i=1,...,5
T_EXERCISE: np.ndarray = np.array([4.0, 5.0, 6.0, 7.0, 8.0])  # T_1,...,T_5
T_COMMON: float = 9.0            # T_e = T_6
N_COMPONENTS: int = 5            # number of Bermudan components

# Time grid
N_STEPS: int = int(T_COMMON / DELTA_T)  # 288 steps
T_GRID: np.ndarray = np.arange(0, N_STEPS + 1) * DELTA_T  # shape (289,)

# Exercise date indices on the time grid
EXERCISE_IDX: List[int] = [int(T_EXERCISE[i] / DELTA_T) for i in range(N_COMPONENTS)]
# = [128, 160, 192, 224, 256]

# Within-factor correlation rho(S^i, X^i) = -0.259
RHO_SW: float = -0.259

# Strike values for swaption pricing
STRIKES: List[float] = [0.01, 0.015, 0.02]
STRIKE_ATM: float = 0.02

# Bermudan swaption strike
BERMUDAN_STRIKE: float = 0.02

# Annual delta (payment frequency)
DELTA_TENOR: float = 1.0  # annual payments

# --- Training Parameters ---
N_PATHS_TOTAL: int = 256        # total MC paths (128 train + 128 test)
N_PATHS_TRAIN: int = 128
N_PATHS_TEST: int = 128

# Inner MC for swaption pricing
N_INNER: int = 2000             # inner Monte Carlo paths per grid point
N_GRID: int = 8                 # grid nodes per dimension (8^3=512 per (i,k))

# Neural network training
N_EPOCHS: int = 30
BATCH_SIZE: int = 64
LR: float = 1e-3
DROPOUT: float = 0.5
N_HIDDEN: int = 4
N_UNITS: int = 32

# CVaR levels
ALPHA_LEVELS: List[float] = [0.2, 0.4, 0.6, 0.8]

# Hedging strategies (S^Max is analytical, no training needed)
STRATEGIES: List[str] = ['S_OS', 'S_I', 'S_IS', 'S_ISM']

# Random seed for reproducibility
SEED: int = 42
SEED_TEST: int = 123

# --- Correlation Matrix Construction ---
def build_correlation_matrix(n_active: int) -> np.ndarray:
    """Build the correlation matrix for the SMBM model for n_active coterminal pairs.
    
    Structure: block matrix [[C_SS, C_SX], [C_XS, C_XX]] where:
    - n = n_active (number of active swap rates / variance states each)
    - Total size: 2*n x 2*n
    - C_SS[i,j] = rho_SS^|i-j| with rho_SS=0.7 (swap-swap correlation)
    - C_XX[i,j] = rho_XX^|i-j| with rho_XX=0.5 (variance-variance correlation)
    - C_SX[i,i] = -0.259 (within-factor S-X correlation)
    - C_SX[i,j] = 0 for i!=j (cross-factor negligible)
    
    Args:
        n_active: Number of active swap rates (also equals number of variance states).
        
    Returns:
        Correlation matrix of shape (2*n_active, 2*n_active).
    """
    n = n_active
    dim = 2 * n
    C = np.zeros((dim, dim))
    
    # S-S block (rows/cols 0..n-1)
    # rho_SS=0.70 ensures the full 10x10 block matrix remains PD for all n in {1..5}
    # (rho_SS=0.80 gives min_eig=-0.044 at n=5, requiring renormalization that
    # distorts the within-factor S-X entries away from the target RHO_SW=-0.259)
    rho_SS = 0.70
    for i in range(n):
        for j in range(n):
            C[i, j] = rho_SS ** abs(i - j)
    
    # X-X block (rows/cols n..2n-1)
    rho_XX = 0.50
    for i in range(n):
        for j in range(n):
            C[n + i, n + j] = rho_XX ** abs(i - j)
    
    # S-X block (within-factor correlation only)
    for i in range(n):
        C[i, n + i] = RHO_SW
        C[n + i, i] = RHO_SW
    
    # Ensure exact symmetry
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    
    # Regularize to ensure positive definiteness
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-8:
        C += (1e-7 - eigvals.min()) * np.eye(dim)
        # Re-normalize diagonal
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
    
    return C


def get_cholesky_factors():
    """Precompute Cholesky factors for all 5 active periods.
    
    Period p has n_active[p] active factors: [5,4,3,2,1] for p=0,...,4.
    
    Returns:
        List of 5 lower Cholesky factors L_p, each of shape (2*n_p, 2*n_p).
    """
    n_active_list = [5, 4, 3, 2, 1]  # active components per period
    chol_factors = []
    for n_act in n_active_list:
        C = build_correlation_matrix(n_act)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # Add regularization
            eigvals = np.linalg.eigvalsh(C)
            eps = max(1e-8, -eigvals.min() + 1e-8)
            C += eps * np.eye(C.shape[0])
            d = np.sqrt(np.diag(C))
            C = C / np.outer(d, d)
            L = np.linalg.cholesky(C)
        chol_factors.append(L)
    return chol_factors


# Period boundaries (start time in years for each period)
PERIOD_BOUNDARIES: List[float] = [0.0, 4.0, 5.0, 6.0, 7.0, 8.0]
