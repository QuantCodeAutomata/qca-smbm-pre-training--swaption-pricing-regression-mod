# SMBM Deep Hedging — Repository Notes

## Project Layout
```
src/
  params.py            # All model constants (OMEGA, XI0, T_EXERCISE, etc.)
  black_scholes.py     # bs_receiver_swaption (Black-76 log-normal formula)
  bootstrap.py         # Coterminal swap bootstrap (discount/annuity factors)
  swaption_pricing.py  # Exp-1: swaption regression via RegularGridInterpolator
data/                  # Saved model .pkl files (swaption_model_i{1-5}_k{0.010,0.015,0.020}.pkl)
exp/                   # Experiment runner scripts
results/               # Output artefacts
tests/                 # Test suite
```

## Key Parameters (src/params.py)
| Symbol      | Value                                      | Meaning                              |
|-------------|--------------------------------------------|--------------------------------------|
| OMEGA       | 0.8403                                     | Vol-of-vol                           |
| KAPPA       | 0.0                                        | Mean reversion (zero under annuity)  |
| S0          | 0.02                                       | ATM initial swap rate                |
| DELTA_T     | 1/32 ≈ 0.03125 yr                         | Simulation time step                 |
| XI0         | [1.132e-4, 1.228e-4, 1.287e-4, 1.344e-4, 1.451e-4] | xi_0^{i,e} per component |
| T_EXERCISE  | [4, 5, 6, 7, 8] yr                        | T_1…T_5 exercise dates               |
| T_COMMON    | 9.0 yr                                     | T_e = T_6 (common maturity)          |
| RHO_SW      | -0.259                                     | Within-factor S–X correlation        |
| STRIKES     | [0.01, 0.015, 0.02]                        | Swaption strikes                     |
| N_GRID      | 8                                          | Grid nodes per dimension (8³=512)    |
| N_INNER     | 2000                                       | Inner MC paths per grid point        |

## Exp-1: Swaption Pricing Regression (src/swaption_pricing.py)
- **Purpose**: pre-train 15 interpolation models v^{i,e,k}(S, X, tau) ≈ E^{A^{i,e}}[(k−S_T)^+]
- **Method**: `scipy.interpolate.RegularGridInterpolator` (linear, bounds_error=False, fill_value=None)
- **Grid**: 3-D (S × X × tau), each axis has N_GRID=8 nodes; 512 points per (i,k) pair
- **Dynamics under annuity measure A^{i,e}**:
  - `dS_t = sigma_t dW_t`  (martingale, no drift)
  - `dX_t = dZ_t`           (Brownian, no drift)
  - `sigma_t = sqrt(xi0_i) * exp((omega/2)*X_t − (omega²/4)*t)`
  - Cholesky correlation: `L = [[1,0],[rho, sqrt(1−rho²)]]`
- **Persistence**: models saved as `.pkl` in `data/`; cached on re-run

### Quick Smoke Test
```bash
cd /workspace/project && python3 -c "
from src.swaption_pricing import compute_swaption_price
p = compute_swaption_price(0.02, 0.0, 4.0, 0.02, 1.132e-04, N_inner=500, seed=42)
print('ATM price (seed=42, N=500):', round(p*1e4, 2), 'bp')
# Expected: ~74.5 bp
"
```

## Python Environment
- Python 3.12.13
- numpy 1.26.4, scipy 1.14.1

## Conventions
- All prices are **annuity-normalised** (no notional or discount factor)
- Basis points (bp) = price × 10 000
- Component indices are **0-based** internally; file names use 1-based (`i{1..5}`)
- `DELTA_T = 1/32` gives 32 steps/year; N_STEPS = 288 for the 9-year horizon
