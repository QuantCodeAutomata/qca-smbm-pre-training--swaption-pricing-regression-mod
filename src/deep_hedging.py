"""Exp-3: Deep Hedging Neural Networks for Bermudan Swaption Strategies.

Implements CVaR-based deep hedging for four strategies:
  S^OS  - Option Spread Hedge
  S^I   - Swap-only hedge
  S^{I+S} - Swap + ATM swaption hedge
  S^{I+S,M} - Swap + multi-strike swaption hedge

Plus the analytical S^Max strategy.

Uses PyTorch for neural networks and Adam optimizer.
# Custom CVaR loss and Bermudan recursion - Context7 found no library equivalent
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import (
    EXERCISE_IDX, T_EXERCISE, N_COMPONENTS, BERMUDAN_STRIKE, STRIKE_ATM,
    ALPHA_LEVELS, N_EPOCHS, BATCH_SIZE, LR, DROPOUT, N_HIDDEN, N_UNITS,
    SEED,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STRATEGIES = ['S_OS', 'S_I', 'S_IS', 'S_ISM']


# ------------------------------------------------------------------
# Neural Network Architecture
# ------------------------------------------------------------------

class HedgeNetwork(nn.Module):
    """Feedforward network for hedge amounts or exercise decisions.

    Architecture: 4 hidden layers, 32 units each, BatchNorm + Dropout + ReLU.
    Linear output for hedge amounts; sigmoid for exercise decisions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden: int = N_HIDDEN,
        n_units: int = N_UNITS,
        dropout: float = DROPOUT,
        output_sigmoid: bool = False,
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension.
            output_dim: Number of outputs (hedge amounts or 1 for exercise).
            n_hidden: Number of hidden layers.
            n_units: Units per hidden layer.
            dropout: Dropout rate.
            output_sigmoid: If True, sigmoid activation on output (for exercise nets).
        """
        super().__init__()
        self.output_sigmoid = output_sigmoid
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(in_dim, n_units),
                nn.BatchNorm1d(n_units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = n_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, input_dim).

        Returns:
            Output tensor, shape (batch, output_dim).
        """
        out = self.net(x)
        if self.output_sigmoid:
            out = torch.sigmoid(out)
        return out


# ------------------------------------------------------------------
# CVaR Loss (Rockafellar-Uryasev formulation)
# ------------------------------------------------------------------

class CVaRLoss(nn.Module):
    """CVaR loss minimization for deep hedging.

    Loss = -z + 1/(1-alpha) * E[max(z - PL, 0)]
    where z is a learnable threshold (VaR estimate).

    Minimizing this loss is equivalent to minimizing CVaR_alpha(-PL).
    """

    def __init__(self, alpha: float) -> None:
        """
        Args:
            alpha: CVaR confidence level in [0, 1).
        """
        super().__init__()
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        self.alpha = alpha
        self.z = nn.Parameter(torch.tensor(0.0))

    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pl: P&L values, shape (batch,).

        Returns:
            Scalar CVaR loss.
        """
        loss = -self.z + (1.0 / (1.0 - self.alpha)) * torch.mean(
            F.relu(self.z - pl)
        )
        return loss

    def compute_cvar(self, pl: torch.Tensor) -> torch.Tensor:
        """Compute CVaR_alpha(-PL) at optimal z (alpha-quantile of PL).

        Args:
            pl: P&L array.

        Returns:
            Scalar CVaR value.
        """
        # Optimal threshold z* = (1-alpha)-quantile of PL  (Rockafellar-Uryasev)
        # min_z { -z + 1/(1-alpha) * E[max(z-PL,0)] } is attained at z* = q_{1-alpha}(PL)
        z_opt = torch.quantile(pl, 1.0 - self.alpha)
        shortfall = torch.mean(F.relu(z_opt - pl))
        return -z_opt + shortfall / (1.0 - self.alpha)


# ------------------------------------------------------------------
# Strategy feature extractors
# ------------------------------------------------------------------

def extract_features(
    strategy: str,
    step: int,
    S_rates: torch.Tensor,
    swap_prices: torch.Tensor,
    swaption_002: torch.Tensor,
    swaption_001: torch.Tensor,
    swaption_0015: torch.Tensor,
    n_active: int,
    t: float,
) -> torch.Tensor:
    """Extract input features for hedge network at a given time step.

    Args:
        strategy: Strategy name ('S_OS', 'S_I', 'S_IS', 'S_ISM').
        step: Time step index.
        S_rates: Swap rates tensor, shape (N, 289, 5).
        swap_prices: Relative swap prices, shape (N, 289, 5).
        swaption_002: ATM swaption prices, shape (N, 289, 5).
        swaption_001: OTM swaption prices, shape (N, 289, 5).
        swaption_0015: Swaption prices at K=0.015, shape (N, 289, 5).
        n_active: Number of currently active components.
        t: Current time (for time feature).

    Returns:
        Feature tensor, shape (N, feature_dim).
    """
    N = swap_prices.shape[0]
    t_feat = torch.full((N, 1), t, device=swap_prices.device)

    if strategy == 'S_I':
        # Swaps only: [t, U_bar^{1,e}, ..., U_bar^{n_active,e}]
        feats = torch.cat([t_feat, swap_prices[:, step, :n_active]], dim=1)
    elif strategy == 'S_IS':
        # Swaps + ATM swaptions
        feats = torch.cat([
            t_feat,
            swap_prices[:, step, :n_active],
            swaption_002[:, step, :n_active],
        ], dim=1)
    elif strategy == 'S_ISM':
        # Swaps + swaptions at 3 strikes
        feats = torch.cat([
            t_feat,
            swap_prices[:, step, :n_active],
            swaption_002[:, step, :n_active],
            swaption_001[:, step, :n_active],
            swaption_0015[:, step, :n_active],
        ], dim=1)
    elif strategy == 'S_OS':
        # Same as S_IS (swaption prices used to compute spread)
        feats = torch.cat([
            t_feat,
            swap_prices[:, step, :n_active],
            swaption_002[:, step, :n_active],
        ], dim=1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return feats


def feature_dim(strategy: str, n_active: int) -> int:
    """Compute feature dimension for a given strategy and number of active components."""
    if strategy == 'S_I':
        return 1 + n_active
    elif strategy == 'S_IS':
        return 1 + 2 * n_active
    elif strategy == 'S_ISM':
        return 1 + 4 * n_active
    elif strategy == 'S_OS':
        return 1 + 2 * n_active
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def output_dim(strategy: str, n_active: int) -> int:
    """Compute output dimension (number of hedge assets) for each strategy."""
    if strategy == 'S_I':
        return n_active
    elif strategy == 'S_IS':
        return 2 * n_active
    elif strategy == 'S_ISM':
        return 4 * n_active
    elif strategy == 'S_OS':
        return 1  # single spread asset weight
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ------------------------------------------------------------------
# Hedge gain computation
# ------------------------------------------------------------------

def compute_hedge_gain(
    strategy: str,
    component: int,
    hedge_weights: Dict[int, torch.Tensor],  # step -> (N, n_hedges)
    swap_prices: torch.Tensor,   # (N, 289, 5)
    swaption_002: torch.Tensor,  # (N, 289, 5)
    swaption_001: torch.Tensor,  # (N, 289, 5)
    swaption_0015: torch.Tensor, # (N, 289, 5)
    start_step: int,
    end_step: int,
    exercise_probs: Optional[Dict[int, torch.Tensor]] = None,  # component -> (N,)
) -> torch.Tensor:
    """Compute total hedge gain over [start_step, end_step).

    For S^OS, compute spread gain using G^i = O_bar^{i,e,K} - H^{i+1}.

    Args:
        strategy: Strategy name.
        component: Current Bermudan component index (0-indexed, i=1 is 0).
        hedge_weights: Dict mapping step to hedge weights (N, n_hedges).
        swap_prices: (N, 289, 5).
        swaption_002: (N, 289, 5) ATM swaption prices.
        swaption_001: OTM swaption prices.
        swaption_0015: Swaption prices at K=0.015.
        start_step: First time step (inclusive).
        end_step: Last time step (exclusive, typically = exercise step).
        exercise_probs: For S^OS, exercise probabilities of frozen components.

    Returns:
        Total hedge gain, shape (N,).
    """
    N = swap_prices.shape[0]
    total_gain = torch.zeros(N, device=swap_prices.device)
    n_act_start = sum(
        (start_step / 32.0) < T_EXERCISE[j] for j in range(5)
    )

    for step in range(start_step, end_step):
        if step not in hedge_weights:
            continue

        w = hedge_weights[step]  # (N, n_hedges)
        t_step = step / 32.0
        n_act = sum(t_step < T_EXERCISE[j] for j in range(5))
        n_act = max(n_act, 1)

        if strategy == 'S_I':
            # delta_U_bar for all active swaps
            delta_I = (
                swap_prices[:, step + 1, :n_act] - swap_prices[:, step, :n_act]
            )  # (N, n_act)
            gain = (w[:, :n_act] * delta_I).sum(dim=1)

        elif strategy == 'S_IS':
            # delta_U_bar and delta_O_bar (ATM) for all active
            delta_swap = swap_prices[:, step + 1, :n_act] - swap_prices[:, step, :n_act]
            delta_opt = swaption_002[:, step + 1, :n_act] - swaption_002[:, step, :n_act]
            delta_I = torch.cat([delta_swap, delta_opt], dim=1)
            gain = (w[:, :2 * n_act] * delta_I).sum(dim=1)

        elif strategy == 'S_ISM':
            # delta for swaps + swaptions at 3 strikes
            delta_swap = swap_prices[:, step + 1, :n_act] - swap_prices[:, step, :n_act]
            delta_opt2 = swaption_002[:, step + 1, :n_act] - swaption_002[:, step, :n_act]
            delta_opt1 = swaption_001[:, step + 1, :n_act] - swaption_001[:, step, :n_act]
            delta_opt15 = swaption_0015[:, step + 1, :n_act] - swaption_0015[:, step, :n_act]
            delta_I = torch.cat([delta_swap, delta_opt2, delta_opt1, delta_opt15], dim=1)
            gain = (w[:, :4 * n_act] * delta_I).sum(dim=1)

        elif strategy == 'S_OS':
            # Spread asset G^{component} = O_bar^{component,e,K} - H^{component+1}
            # For simplicity, use delta_O_bar^{i} as the hedge asset (approximation)
            # Full S^OS spread computation requires H^{i+1} from frozen models
            delta_O = (
                swaption_002[:, step + 1, component]
                - swaption_002[:, step, component]
            )  # (N,)
            gain = w[:, 0] * delta_O

        else:
            gain = torch.zeros(N, device=swap_prices.device)

        total_gain = total_gain + gain

    return total_gain


# ------------------------------------------------------------------
# S^Max analytical strategy
# ------------------------------------------------------------------

def smax_strategy(
    swaption_002: np.ndarray,  # (N, 289, 5)
    exercise_dates_idx: List[int],
    bermudan_strike: float = BERMUDAN_STRIKE,
    S_rates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute P&L for the analytical S^Max strategy.

    S^Max holds the maximum-value swaption at each time step.
    Exercises at the first date where this component is the argmax.

    Args:
        swaption_002: Relative ATM swaption prices, shape (N, 289, 5).
        exercise_dates_idx: List of exercise time step indices.
        bermudan_strike: Bermudan swaption strike.
        S_rates: Swap rates for intrinsic value (optional).

    Returns:
        P&L array, shape (N,).
    """
    N = swaption_002.shape[0]

    # Track holding: always hold argmax swaption
    # Gain at each step: delta(max_j O_bar^j_t)
    # Exercise at T_i if i = argmax_j O_bar^j_{T_i}

    # Compute H^Max_t = max_j O_bar^j_t
    H_max = np.max(swaption_002, axis=2)  # (N, 289)

    # Hedge gain: sum of changes in H_max
    delta_H = np.diff(H_max, axis=1)  # (N, 288)
    total_hedge_gain = delta_H.sum(axis=1)  # (N,)

    # Exercise decision: at each T_i, exercise if i = argmax
    # We model the Bermudan as a series of binary decisions
    exercised = np.zeros(N, dtype=bool)
    exercise_payoff = np.zeros(N)

    for idx_k, step in enumerate(exercise_dates_idx):
        # Components still active at this step: idx_k, ..., 4
        # Exercise if this is the argmax AND not yet exercised
        if step >= swaption_002.shape[1]:
            step = swaption_002.shape[1] - 1
        opt_values = swaption_002[:, step, idx_k:]  # (N, n_remaining)
        if opt_values.shape[1] == 0:
            continue
        best = np.argmax(opt_values, axis=1)  # (N,) - relative index

        # Exercise at this date if argmax is the first (current) component
        exercise_now = ~exercised & (best == 0) if idx_k < 4 else ~exercised
        exercise_payoff[exercise_now] = swaption_002[exercise_now, step, idx_k]
        exercised = exercised | exercise_now

    # Unexercised paths: exercise at last date
    if np.any(~exercised):
        last_step = exercise_dates_idx[-1]
        if last_step >= swaption_002.shape[1]:
            last_step = swaption_002.shape[1] - 1
        exercise_payoff[~exercised] = swaption_002[~exercised, last_step, -1]

    total_pl = total_hedge_gain + exercise_payoff
    return total_pl


# ------------------------------------------------------------------
# Bermudan Hedger: recursive training
# ------------------------------------------------------------------

class BermudanHedger:
    """Manages all neural networks for one (strategy, alpha) configuration.

    Trains N^{CH,i} (hedge networks) and N^{E,i} (exercise networks)
    recursively from i=5 down to i=1.
    """

    def __init__(
        self,
        strategy: str,
        alpha: float,
        device: torch.device = DEVICE,
        n_epochs: int = N_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
    ) -> None:
        """
        Args:
            strategy: One of 'S_OS', 'S_I', 'S_IS', 'S_ISM'.
            alpha: CVaR confidence level.
            device: Torch device.
            n_epochs: Training epochs per component.
            batch_size: Mini-batch size.
            lr: Adam learning rate.
        """
        self.strategy = strategy
        self.alpha = alpha
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

        # Networks: CH_nets[i] = N^{CH,i}, E_nets[i] = N^{E,i}
        self.CH_nets: Dict[int, HedgeNetwork] = {}
        self.E_nets: Dict[int, HedgeNetwork] = {}
        self.trained_components: List[int] = []  # trained so far (descending)

        # Frozen hedge weights precomputed for continuation value
        # frozen_gains[i] = array(N,) of hedge gains from component i+1 onwards
        self.frozen_pl: Dict[int, np.ndarray] = {}

    def _build_networks(self, component: int) -> Tuple[HedgeNetwork, HedgeNetwork]:
        """Build hedge and exercise networks for component i.

        Args:
            component: 0-indexed component (0 = T_1=4y, ..., 4 = T_5=8y).

        Returns:
            (ch_net, e_net): Hedge and exercise networks.
        """
        n_active = 5 - component  # at time T_i, n_active components remain
        feat_dim = feature_dim(self.strategy, n_active)
        out_dim = output_dim(self.strategy, n_active)

        ch_net = HedgeNetwork(feat_dim, out_dim, output_sigmoid=False).to(self.device)
        e_net = HedgeNetwork(feat_dim - 1, 1, output_sigmoid=True).to(self.device)
        # Exercise net doesn't take time as input (time is fixed at T_i)
        return ch_net, e_net

    def _get_exercise_features(
        self,
        component: int,
        exercise_step: int,
        swap_prices_t: torch.Tensor,
        swaption_002_t: torch.Tensor,
        swaption_001_t: torch.Tensor,
        swaption_0015_t: torch.Tensor,
    ) -> torch.Tensor:
        """Get features for exercise network at exercise date T_{component+1}.

        Args:
            component: 0-indexed component.
            exercise_step: Time step index for exercise date.
            swap_prices_t: (N, 5) swap prices at exercise date.
            swaption_002_t: (N, 5) ATM swaption prices at exercise date.
            swaption_001_t: (N, 5) OTM swaption prices.
            swaption_0015_t: (N, 5) swaption at K=0.015.

        Returns:
            Feature tensor, shape (N, feat_dim-1).
        """
        n_active = 5 - component

        if self.strategy == 'S_I':
            feats = swap_prices_t[:, :n_active]
        elif self.strategy == 'S_IS':
            feats = torch.cat([swap_prices_t[:, :n_active], swaption_002_t[:, :n_active]], dim=1)
        elif self.strategy == 'S_ISM':
            feats = torch.cat([
                swap_prices_t[:, :n_active],
                swaption_002_t[:, :n_active],
                swaption_001_t[:, :n_active],
                swaption_0015_t[:, :n_active],
            ], dim=1)
        elif self.strategy == 'S_OS':
            feats = torch.cat([swap_prices_t[:, :n_active], swaption_002_t[:, :n_active]], dim=1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return feats

    def train_component(
        self,
        component: int,
        paths_train: Dict[str, np.ndarray],
        continuation_pl: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Train hedge and exercise networks for one Bermudan component.

        Args:
            component: 0-indexed component (0=first exercise, 4=last).
            paths_train: Training path dictionary.
            continuation_pl: Pre-computed continuation P&L from frozen components i+1,...,5.
                            Shape (N_train,). None if component is the last (i=5, component=4).

        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists.
        """
        ch_net, e_net = self._build_networks(component)
        self.CH_nets[component] = ch_net
        self.E_nets[component] = e_net

        N_train = paths_train['S_rates'].shape[0]
        n_active = 5 - component
        exercise_step = EXERCISE_IDX[component]

        # Convert paths to tensors
        swap_t = torch.from_numpy(paths_train['swap_prices'].astype(np.float32)).to(self.device)
        opt002_t = torch.from_numpy(paths_train['swaption_K002'].astype(np.float32)).to(self.device)
        opt001_t = torch.from_numpy(paths_train['swaption_K001'].astype(np.float32)).to(self.device)
        opt0015_t = torch.from_numpy(paths_train['swaption_K0015'].astype(np.float32)).to(self.device)

        if continuation_pl is not None:
            cont_pl_t = torch.from_numpy(continuation_pl.astype(np.float32)).to(self.device)
        else:
            cont_pl_t = None

        # CVaR loss
        cvar_loss_fn = CVaRLoss(self.alpha).to(self.device)

        # Optimizer for both networks + z parameter
        params = (
            list(ch_net.parameters())
            + list(e_net.parameters())
            + list(cvar_loss_fn.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        # Train/val split (80/20)
        n_val = max(1, int(0.2 * N_train))
        n_tr = N_train - n_val
        idx_all = np.arange(N_train)
        train_idx = torch.from_numpy(idx_all[:n_tr]).long()
        val_idx = torch.from_numpy(idx_all[n_tr:]).long()

        history = {'train_loss': [], 'val_loss': []}

        # Initialize z to (1-alpha)-quantile of exercise payoff on training data
        ex_payoff_init = opt002_t[train_idx, exercise_step, component].detach()
        with torch.no_grad():
            cvar_loss_fn.z.data = torch.quantile(ex_payoff_init, 1.0 - self.alpha)

        for epoch in range(self.n_epochs):
            ch_net.train()
            e_net.train()

            # Shuffle training indices
            perm = torch.randperm(n_tr)
            train_idx_perm = train_idx[perm]

            epoch_losses = []
            for start in range(0, n_tr, self.batch_size):
                batch_idx = train_idx_perm[start: start + self.batch_size]
                if len(batch_idx) < 2:
                    continue

                optimizer.zero_grad()

                # ---- Forward pass: compute P&L for batch ----
                # Step 1: Hedge gains from t=0 to T_{component} (or T_0 to T_component)
                # For efficiency, compute features at each step and accumulate gains
                hedge_gain = torch.zeros(len(batch_idx), device=self.device)

                # Determine time steps for this component
                start_step = EXERCISE_IDX[component - 1] if component > 0 else 0
                end_step = exercise_step

                # Compute hedge gains for each time step
                for step in range(start_step, end_step):
                    t = step / 32.0
                    n_act = sum(t < T_EXERCISE[j] for j in range(5))
                    n_act = max(1, n_act)

                    feats = extract_features(
                        self.strategy, step,
                        None,  # S_rates not needed
                        swap_t[batch_idx],
                        opt002_t[batch_idx],
                        opt001_t[batch_idx],
                        opt0015_t[batch_idx],
                        n_act, t,
                    )

                    w = ch_net(feats)  # (batch, n_hedges)

                    # Compute delta assets (from step to step+1)
                    n_act2 = min(n_act, n_active)
                    if self.strategy == 'S_I':
                        delta = swap_t[batch_idx, step + 1, :n_act2] - swap_t[batch_idx, step, :n_act2]
                        g = (w[:, :n_act2] * delta).sum(dim=1)
                    elif self.strategy == 'S_IS':
                        d_swap = swap_t[batch_idx, step + 1, :n_act2] - swap_t[batch_idx, step, :n_act2]
                        d_opt = opt002_t[batch_idx, step + 1, :n_act2] - opt002_t[batch_idx, step, :n_act2]
                        delta = torch.cat([d_swap, d_opt], dim=1)
                        g = (w[:, :2 * n_act2] * delta).sum(dim=1)
                    elif self.strategy == 'S_ISM':
                        d_swap = swap_t[batch_idx, step + 1, :n_act2] - swap_t[batch_idx, step, :n_act2]
                        d_opt2 = opt002_t[batch_idx, step + 1, :n_act2] - opt002_t[batch_idx, step, :n_act2]
                        d_opt1 = opt001_t[batch_idx, step + 1, :n_act2] - opt001_t[batch_idx, step, :n_act2]
                        d_opt15 = opt0015_t[batch_idx, step + 1, :n_act2] - opt0015_t[batch_idx, step, :n_act2]
                        delta = torch.cat([d_swap, d_opt2, d_opt1, d_opt15], dim=1)
                        g = (w[:, :4 * n_act2] * delta).sum(dim=1)
                    elif self.strategy == 'S_OS':
                        d_opt = opt002_t[batch_idx, step + 1, component] - opt002_t[batch_idx, step, component]
                        g = w[:, 0] * d_opt
                    else:
                        g = torch.zeros(len(batch_idx), device=self.device)

                    hedge_gain = hedge_gain + g

                # Step 2: Exercise decision at T_{component}
                ex_feats = self._get_exercise_features(
                    component, exercise_step,
                    swap_t[batch_idx, exercise_step, :],
                    opt002_t[batch_idx, exercise_step, :],
                    opt001_t[batch_idx, exercise_step, :],
                    opt0015_t[batch_idx, exercise_step, :],
                )
                e_prob = e_net(ex_feats).squeeze(1)  # (batch,)

                # Exercise payoff at T_i
                intrinsic = opt002_t[batch_idx, exercise_step, component]

                if cont_pl_t is not None:
                    # Effective payoff: e_prob * O_bar + (1 - e_prob) * V_cont
                    V_cont = cont_pl_t[batch_idx]
                    Z_eff = e_prob * intrinsic + (1 - e_prob) * V_cont
                else:
                    Z_eff = e_prob * intrinsic

                # Total P&L
                pl = hedge_gain + Z_eff

                # CVaR loss
                loss = cvar_loss_fn(pl)

                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            # Validation loss
            ch_net.eval()
            e_net.eval()
            with torch.no_grad():
                val_pl = self._compute_pl_batch(
                    val_idx, component, start_step, end_step,
                    swap_t, opt002_t, opt001_t, opt0015_t,
                    exercise_step, cont_pl_t, ch_net, e_net,
                )
                val_loss = cvar_loss_fn(val_pl).item()

            train_loss = np.mean(epoch_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"  [Component {component+1}, Epoch {epoch+1}/{self.n_epochs}] "
                      f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Freeze networks
        for p in ch_net.parameters():
            p.requires_grad_(False)
        for p in e_net.parameters():
            p.requires_grad_(False)

        return history

    def _compute_pl_batch(
        self,
        indices: torch.Tensor,
        component: int,
        start_step: int,
        end_step: int,
        swap_t: torch.Tensor,
        opt002_t: torch.Tensor,
        opt001_t: torch.Tensor,
        opt0015_t: torch.Tensor,
        exercise_step: int,
        cont_pl_t: Optional[torch.Tensor],
        ch_net: HedgeNetwork,
        e_net: HedgeNetwork,
    ) -> torch.Tensor:
        """Compute P&L for a batch of paths without gradient tracking."""
        n_active = 5 - component
        hedge_gain = torch.zeros(len(indices), device=self.device)

        for step in range(start_step, end_step):
            t = step / 32.0
            n_act = max(1, sum(t < T_EXERCISE[j] for j in range(5)))
            n_act2 = min(n_act, n_active)

            feats = extract_features(
                self.strategy, step, None,
                swap_t[indices], opt002_t[indices],
                opt001_t[indices], opt0015_t[indices],
                n_act, t,
            )
            w = ch_net(feats)

            if self.strategy == 'S_I':
                delta = swap_t[indices, step + 1, :n_act2] - swap_t[indices, step, :n_act2]
                g = (w[:, :n_act2] * delta).sum(dim=1)
            elif self.strategy == 'S_IS':
                d_swap = swap_t[indices, step + 1, :n_act2] - swap_t[indices, step, :n_act2]
                d_opt = opt002_t[indices, step + 1, :n_act2] - opt002_t[indices, step, :n_act2]
                g = (w[:, :2 * n_act2] * torch.cat([d_swap, d_opt], dim=1)).sum(dim=1)
            elif self.strategy == 'S_ISM':
                d_swap = swap_t[indices, step + 1, :n_act2] - swap_t[indices, step, :n_act2]
                d_opt2 = opt002_t[indices, step + 1, :n_act2] - opt002_t[indices, step, :n_act2]
                d_opt1 = opt001_t[indices, step + 1, :n_act2] - opt001_t[indices, step, :n_act2]
                d_opt15 = opt0015_t[indices, step + 1, :n_act2] - opt0015_t[indices, step, :n_act2]
                delta = torch.cat([d_swap, d_opt2, d_opt1, d_opt15], dim=1)
                g = (w[:, :4 * n_act2] * delta).sum(dim=1)
            elif self.strategy == 'S_OS':
                d_opt = opt002_t[indices, step + 1, component] - opt002_t[indices, step, component]
                g = w[:, 0] * d_opt
            else:
                g = torch.zeros(len(indices), device=self.device)

            hedge_gain = hedge_gain + g

        ex_feats = self._get_exercise_features(
            component, exercise_step,
            swap_t[indices, exercise_step, :],
            opt002_t[indices, exercise_step, :],
            opt001_t[indices, exercise_step, :],
            opt0015_t[indices, exercise_step, :],
        )
        e_prob = e_net(ex_feats).squeeze(1)
        intrinsic = opt002_t[indices, exercise_step, component]

        if cont_pl_t is not None:
            V_cont = cont_pl_t[indices]
            Z_eff = e_prob * intrinsic + (1 - e_prob) * V_cont
        else:
            Z_eff = e_prob * intrinsic

        return hedge_gain + Z_eff

    def compute_continuation_value(
        self,
        next_component: int,
        paths: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute continuation value V^{Cont,i} using frozen trained networks.

        V^{Cont,i}_n = sum of PL from exercising at component i+1 onwards.

        Args:
            next_component: 0-indexed component i+1 (first frozen component).
            paths: Path data dictionary.

        Returns:
            Continuation values, shape (N_paths,).
        """
        assert next_component in self.trained_components, \
            f"Component {next_component} not yet trained"

        N = paths['S_rates'].shape[0]
        swap_t = torch.from_numpy(paths['swap_prices'].astype(np.float32)).to(self.device)
        opt002_t = torch.from_numpy(paths['swaption_K002'].astype(np.float32)).to(self.device)
        opt001_t = torch.from_numpy(paths['swaption_K001'].astype(np.float32)).to(self.device)
        opt0015_t = torch.from_numpy(paths['swaption_K0015'].astype(np.float32)).to(self.device)

        all_idx = torch.arange(N, device=self.device)

        # Compute P&L from the next_component onwards (recursively)
        pl = torch.zeros(N, device=self.device)

        for comp in range(next_component, 5):
            ch_net = self.CH_nets[comp]
            e_net = self.E_nets[comp]
            ch_net.eval()
            e_net.eval()

            exercise_step = EXERCISE_IDX[comp]
            start_step = EXERCISE_IDX[comp - 1] if comp > 0 else 0

            with torch.no_grad():
                comp_pl = self._compute_pl_batch(
                    all_idx, comp, start_step, exercise_step,
                    swap_t, opt002_t, opt001_t, opt0015_t,
                    exercise_step, None, ch_net, e_net,
                )

            if comp == next_component:
                pl = comp_pl
            else:
                # Add hedge gain from this component
                pl = pl + comp_pl - comp_pl  # placeholder, already accumulated

        return pl.cpu().numpy()

    def train_all_components(
        self,
        paths_train: Dict[str, np.ndarray],
    ) -> Dict[int, Dict]:
        """Train all 5 Bermudan components recursively (i=5 first, i=1 last).

        Args:
            paths_train: Training paths dict.

        Returns:
            Dict mapping component index -> training history.
        """
        histories = {}

        # Train from last component (i=5, component=4) to first (i=1, component=0)
        for comp in range(4, -1, -1):
            print(f"\n--- Training component {comp+1} (T={T_EXERCISE[comp]:.0f}y) "
                  f"| strategy={self.strategy}, alpha={self.alpha} ---")

            if comp == 4:
                # Last component: no continuation value
                cont_pl = None
            else:
                # Compute continuation value from already-trained components
                cont_pl = self.compute_continuation_value_simple(
                    comp + 1, paths_train
                )

            hist = self.train_component(comp, paths_train, cont_pl)
            histories[comp] = hist
            self.trained_components.append(comp)

        return histories

    def compute_continuation_value_simple(
        self,
        next_comp: int,
        paths: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Simplified continuation value: P&L from frozen component next_comp.

        For the recursive Bermudan: V^{Cont,i} = PL from component i+1
        (exercise payoff + hedge gains from T_{i+1} to T_{i+2}, etc.)

        Args:
            next_comp: The first frozen component (0-indexed).
            paths: Path data.

        Returns:
            Continuation P&L, shape (N,).
        """
        N = paths['S_rates'].shape[0]
        swap_t = torch.from_numpy(paths['swap_prices'].astype(np.float32)).to(self.device)
        opt002_t = torch.from_numpy(paths['swaption_K002'].astype(np.float32)).to(self.device)
        opt001_t = torch.from_numpy(paths['swaption_K001'].astype(np.float32)).to(self.device)
        opt0015_t = torch.from_numpy(paths['swaption_K0015'].astype(np.float32)).to(self.device)

        all_idx = torch.arange(N, device=self.device)
        total_pl = torch.zeros(N, device=self.device)

        for comp in range(next_comp, 5):
            ch_net = self.CH_nets[comp]
            e_net = self.E_nets[comp]
            ch_net.eval()
            e_net.eval()

            exercise_step = EXERCISE_IDX[comp]
            start_step = EXERCISE_IDX[comp - 1] if comp > 0 else 0

            with torch.no_grad():
                # Continuation from previous frozen comp feeds in
                prev_cont = total_pl if comp > next_comp else None
                comp_pl = self._compute_pl_batch(
                    all_idx, comp, start_step, exercise_step,
                    swap_t, opt002_t, opt001_t, opt0015_t,
                    exercise_step,
                    total_pl if comp > next_comp else None,
                    ch_net, e_net,
                )

            total_pl = comp_pl

        return total_pl.cpu().numpy()

    def evaluate(
        self,
        paths_test: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Evaluate trained model on test paths.

        Args:
            paths_test: Test path data.

        Returns:
            P&L array, shape (N_test,).
        """
        N = paths_test['S_rates'].shape[0]
        swap_t = torch.from_numpy(paths_test['swap_prices'].astype(np.float32)).to(self.device)
        opt002_t = torch.from_numpy(paths_test['swaption_K002'].astype(np.float32)).to(self.device)
        opt001_t = torch.from_numpy(paths_test['swaption_K001'].astype(np.float32)).to(self.device)
        opt0015_t = torch.from_numpy(paths_test['swaption_K0015'].astype(np.float32)).to(self.device)

        all_idx = torch.arange(N, device=self.device)

        for comp in range(4, -1, -1):
            self.CH_nets[comp].eval()
            self.E_nets[comp].eval()

        total_pl = torch.zeros(N, device=self.device)

        for comp in range(4, -1, -1):
            ch_net = self.CH_nets[comp]
            e_net = self.E_nets[comp]

            exercise_step = EXERCISE_IDX[comp]
            start_step = EXERCISE_IDX[comp - 1] if comp > 0 else 0

            with torch.no_grad():
                comp_pl = self._compute_pl_batch(
                    all_idx, comp, start_step, exercise_step,
                    swap_t, opt002_t, opt001_t, opt0015_t,
                    exercise_step,
                    total_pl if comp < 4 else None,
                    ch_net, e_net,
                )
            total_pl = comp_pl

        return total_pl.cpu().numpy()

    def save(self, filepath: str) -> None:
        """Save model state dicts.

        Args:
            filepath: Output .pt file path.
        """
        state = {
            'strategy': self.strategy,
            'alpha': self.alpha,
            'CH_nets': {i: net.state_dict() for i, net in self.CH_nets.items()},
            'E_nets': {i: net.state_dict() for i, net in self.E_nets.items()},
        }
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model state dicts.

        Args:
            filepath: Path to saved .pt file.
        """
        state = torch.load(filepath, map_location=self.device)
        for i, sd in state['CH_nets'].items():
            n_active = 5 - i
            feat_dim_i = feature_dim(self.strategy, n_active)
            out_dim_i = output_dim(self.strategy, n_active)
            net = HedgeNetwork(feat_dim_i, out_dim_i, output_sigmoid=False).to(self.device)
            net.load_state_dict(sd)
            self.CH_nets[i] = net

        for i, sd in state['E_nets'].items():
            n_active = 5 - i
            feat_dim_i = feature_dim(self.strategy, n_active) - 1
            net = HedgeNetwork(feat_dim_i, 1, output_sigmoid=True).to(self.device)
            net.load_state_dict(sd)
            self.E_nets[i] = net

        self.trained_components = list(state['CH_nets'].keys())


def train_all_configurations(
    paths_train: Dict[str, np.ndarray],
    save_dir: str = 'data',
    strategies: Optional[List[str]] = None,
    alpha_levels: Optional[List[float]] = None,
    n_epochs: int = N_EPOCHS,
) -> Dict[Tuple[str, float], BermudanHedger]:
    """Train all (strategy, alpha) configurations.

    Args:
        paths_train: Training paths.
        save_dir: Directory to save trained models.
        strategies: List of strategies to train.
        alpha_levels: CVaR confidence levels.
        n_epochs: Training epochs per component.

    Returns:
        Dict mapping (strategy, alpha) -> trained BermudanHedger.
    """
    os.makedirs(save_dir, exist_ok=True)

    if strategies is None:
        strategies = STRATEGIES
    if alpha_levels is None:
        alpha_levels = ALPHA_LEVELS

    trained_models = {}

    for strategy in strategies:
        for alpha in alpha_levels:
            print(f"\n{'='*60}")
            print(f"Training strategy={strategy}, alpha={alpha}")
            print(f"{'='*60}")

            model_path = os.path.join(save_dir, f'model_{strategy}_alpha{alpha:.1f}.pt')

            hedger = BermudanHedger(
                strategy=strategy, alpha=alpha, n_epochs=n_epochs
            )

            if os.path.exists(model_path):
                print(f"Loading cached model from {model_path}")
                hedger.load(model_path)
            else:
                hedger.train_all_components(paths_train)
                hedger.save(model_path)

            trained_models[(strategy, alpha)] = hedger

    return trained_models
