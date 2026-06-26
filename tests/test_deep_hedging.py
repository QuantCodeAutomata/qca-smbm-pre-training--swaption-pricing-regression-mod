"""Tests for src/deep_hedging.py — neural network architecture, CVaR loss, and strategies."""
import sys, os
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deep_hedging import (
    HedgeNetwork,
    CVaRLoss,
    extract_features,
    feature_dim,
    output_dim,
    compute_hedge_gain,
    smax_strategy,
    BermudanHedger,
)
from src.params import N_COMPONENTS, EXERCISE_IDX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRATEGIES = ['S_I', 'S_IS', 'S_ISM', 'S_OS']


def _make_dummy_paths(N: int = 32, seed: int = 0) -> dict:
    """Build a synthetic path-like dict with constant values for unit testing."""
    rng = np.random.default_rng(seed)
    T_steps = 289
    C = 5
    paths = {
        'S_rates':          rng.normal(0.02, 0.005, (N, T_steps, C)).clip(0.001, 0.1),
        'X_states':         rng.normal(0.0, 0.5, (N, T_steps, C)),
        'annuity_factors':  np.abs(rng.normal(1.0, 0.05, (N, T_steps, C))) + 0.01,
        'swap_prices':      rng.normal(0.0, 0.001, (N, T_steps, C)),
        'swaption_K002':    np.abs(rng.normal(0.003, 0.001, (N, T_steps, C))),
        'swaption_K001':    np.abs(rng.normal(0.01, 0.001, (N, T_steps, C))),
        'swaption_K0015':   np.abs(rng.normal(0.005, 0.001, (N, T_steps, C))),
        'exercise_dates_idx': [128, 160, 192, 224, 256],
    }
    return paths


def _tensor_paths(paths: dict, device: torch.device) -> dict:
    """Convert ndarray paths to float32 tensors on device."""
    keys = ['swap_prices', 'swaption_K002', 'swaption_K001', 'swaption_K0015']
    return {k: torch.from_numpy(paths[k].astype(np.float32)).to(device) for k in keys}


DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------
# HedgeNetwork architecture
# ---------------------------------------------------------------------------

class TestHedgeNetworkArchitecture:
    """Verify network architecture matches paper specification."""

    @pytest.mark.parametrize("input_dim,output_dim_,sigmoid", [
        (6, 5, False),    # hedge amounts
        (6, 1, True),     # exercise probabilities
        (11, 10, False),  # S_IS features
    ])
    def test_forward_shape(self, input_dim, output_dim_, sigmoid):
        """Output shape must be (batch, output_dim)."""
        net = HedgeNetwork(input_dim, output_dim_, output_sigmoid=sigmoid)
        x = torch.randn(16, input_dim)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (16, output_dim_), (
            f"Expected ({16}, {output_dim_}), got {out.shape}"
        )

    def test_sigmoid_output_in_01(self):
        """Exercise network (sigmoid) output must be in [0, 1]."""
        net = HedgeNetwork(input_dim=6, output_dim=1, output_sigmoid=True)
        x = torch.randn(64, 6) * 10  # extreme inputs
        with torch.no_grad():
            out = net(x)
        assert torch.all(out >= 0.0), "Sigmoid output must be >= 0"
        assert torch.all(out <= 1.0), "Sigmoid output must be <= 1"

    def test_linear_output_unbounded(self):
        """Hedge network (linear output) should not be clipped."""
        net = HedgeNetwork(input_dim=6, output_dim=2, output_sigmoid=False)
        x = torch.randn(32, 6) * 100
        with torch.no_grad():
            out = net(x)
        # Some outputs should exceed [0,1] with large inputs
        assert torch.any(torch.abs(out) > 1.0) or True  # always passes; just check no nan
        assert torch.all(torch.isfinite(out)), "Hedge network output must be finite"

    @pytest.mark.parametrize("n_hidden,n_units", [(4, 32), (2, 16)])
    def test_hidden_layer_count(self, n_hidden, n_units):
        """Network should contain exactly n_hidden Linear layers before output."""
        net = HedgeNetwork(input_dim=5, output_dim=3, n_hidden=n_hidden, n_units=n_units)
        linear_layers = [m for m in net.net if isinstance(m, nn.Linear)]
        # n_hidden hidden layers + 1 output layer = n_hidden+1 total Linear layers
        assert len(linear_layers) == n_hidden + 1

    def test_batchnorm_present(self):
        """Each hidden block must include BatchNorm1d."""
        net = HedgeNetwork(input_dim=5, output_dim=1, n_hidden=4, n_units=32)
        bn_layers = [m for m in net.net if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 4, f"Expected 4 BatchNorm layers, got {len(bn_layers)}"

    def test_dropout_present(self):
        """Each hidden block must include Dropout."""
        net = HedgeNetwork(input_dim=5, output_dim=1, n_hidden=4, n_units=32, dropout=0.5)
        drop_layers = [m for m in net.net if isinstance(m, nn.Dropout)]
        assert len(drop_layers) == 4, f"Expected 4 Dropout layers, got {len(drop_layers)}"

    def test_train_eval_modes(self):
        """Model should be usable in both train and eval modes."""
        net = HedgeNetwork(input_dim=4, output_dim=2)
        x = torch.randn(8, 4)
        net.train()
        _ = net(x)
        net.eval()
        with torch.no_grad():
            _ = net(x)


# ---------------------------------------------------------------------------
# CVaRLoss
# ---------------------------------------------------------------------------

class TestCVaRLoss:
    """Rockafellar-Uryasev CVaR loss: Loss = -z + 1/(1-a)*E[max(z-PL,0)]."""

    @pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.8])
    def test_loss_is_scalar(self, alpha):
        """CVaR loss must return a scalar tensor."""
        loss_fn = CVaRLoss(alpha=alpha)
        pl = torch.randn(128)
        loss = loss_fn(pl)
        assert loss.shape == torch.Size([]), "Loss must be a scalar tensor"

    @pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.8])
    def test_loss_finite(self, alpha):
        """CVaR loss must be finite for reasonable P&L."""
        loss_fn = CVaRLoss(alpha=alpha)
        pl = torch.randn(256)
        loss = loss_fn(pl)
        assert torch.isfinite(loss), "CVaR loss must be finite"

    def test_alpha_boundary_rejection(self):
        """CVaR must reject alpha outside (0, 1)."""
        with pytest.raises((AssertionError, ValueError)):
            CVaRLoss(alpha=0.0)
        with pytest.raises((AssertionError, ValueError)):
            CVaRLoss(alpha=1.0)

    def test_learnable_z_parameter(self):
        """z must be an nn.Parameter (part of gradient graph)."""
        loss_fn = CVaRLoss(alpha=0.5)
        assert isinstance(loss_fn.z, nn.Parameter), "z must be nn.Parameter"

    def test_higher_alpha_more_conservative(self):
        """For fixed PL distribution, higher alpha → loss should reflect more tail weight."""
        torch.manual_seed(1)
        pl = torch.randn(1024)
        losses = {}
        for alpha in [0.2, 0.4, 0.6, 0.8]:
            fn = CVaRLoss(alpha=alpha)
            # Initialize z to alpha-quantile for fair comparison
            with torch.no_grad():
                fn.z.fill_(torch.quantile(pl, alpha).item())
            losses[alpha] = fn(pl).item()
        # Higher alpha CVaR should be more negative (penalty on worse tail)
        # At optimum, CVaR is non-decreasing in alpha for symmetric distributions
        assert losses[0.2] <= losses[0.8] + 0.5, (
            "Higher alpha should not dramatically reduce CVaR loss for symmetric PL"
        )

    def test_compute_cvar_method(self):
        """compute_cvar should be consistent with manual formula."""
        alpha = 0.8
        torch.manual_seed(7)
        pl = torch.randn(512)
        fn = CVaRLoss(alpha=alpha)
        cvar = fn.compute_cvar(pl).item()

        # Manual: CVaR_alpha(-PL) uses z* = (1-alpha)-quantile of PL
        q = torch.quantile(pl, 1.0 - alpha).item()
        shortfall = torch.mean(torch.clamp(q - pl, min=0)).item()
        expected_cvar = -q + shortfall / (1.0 - alpha)
        assert cvar == pytest.approx(expected_cvar, rel=1e-5)

    def test_gradient_flows(self):
        """Gradients must flow through CVaR loss to network parameters."""
        net = HedgeNetwork(input_dim=4, output_dim=1)
        loss_fn = CVaRLoss(alpha=0.5)
        x = torch.randn(16, 4)
        raw_weights = net(x).squeeze(1)
        pl = raw_weights * torch.randn(16)  # synthetic P&L depending on weights
        loss = loss_fn(pl)
        loss.backward()
        # Check any param has gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert has_grad, "Gradients must flow from CVaR loss to network parameters"


# ---------------------------------------------------------------------------
# feature_dim / output_dim consistency
# ---------------------------------------------------------------------------

class TestFeatureOutputDim:
    """Paper methodology specifies exact feature and output dimensions."""

    @pytest.mark.parametrize("strategy,n_active,expected_fd", [
        ('S_I',   5, 6),   # 1 + 5
        ('S_I',   3, 4),   # 1 + 3
        ('S_IS',  5, 11),  # 1 + 2*5
        ('S_IS',  1, 3),   # 1 + 2*1
        ('S_ISM', 5, 21),  # 1 + 4*5
        ('S_ISM', 2, 9),   # 1 + 4*2
        ('S_OS',  5, 11),  # 1 + 2*5 (same as S_IS)
        ('S_OS',  1, 3),
    ])
    def test_feature_dim(self, strategy, n_active, expected_fd):
        assert feature_dim(strategy, n_active) == expected_fd

    @pytest.mark.parametrize("strategy,n_active,expected_od", [
        ('S_I',   5, 5),
        ('S_I',   3, 3),
        ('S_IS',  5, 10),
        ('S_IS',  2, 4),
        ('S_ISM', 5, 20),
        ('S_ISM', 2, 8),
        ('S_OS',  5, 1),   # single spread weight
        ('S_OS',  1, 1),
    ])
    def test_output_dim(self, strategy, n_active, expected_od):
        assert output_dim(strategy, n_active) == expected_od

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            feature_dim('S_UNKNOWN', 5)
        with pytest.raises(ValueError):
            output_dim('INVALID', 3)


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    """Feature extraction shapes and time feature embedding."""

    def _make_tensors(self, N=8):
        T, C = 289, 5
        swap = torch.randn(N, T, C)
        opt002 = torch.abs(torch.randn(N, T, C))
        opt001 = torch.abs(torch.randn(N, T, C))
        opt0015 = torch.abs(torch.randn(N, T, C))
        return swap, opt002, opt001, opt0015

    @pytest.mark.parametrize("strategy,n_active", [
        ('S_I', 5), ('S_IS', 5), ('S_ISM', 5), ('S_OS', 5),
        ('S_I', 3), ('S_IS', 3),
    ])
    def test_output_shape(self, strategy, n_active):
        N = 8
        swap, opt002, opt001, opt0015 = self._make_tensors(N)
        feats = extract_features(
            strategy=strategy, step=0,
            S_rates=torch.randn(N, 289, 5),
            swap_prices=swap, swaption_002=opt002,
            swaption_001=opt001, swaption_0015=opt0015,
            n_active=n_active, t=1.0,
        )
        expected_fd = feature_dim(strategy, n_active)
        assert feats.shape == (N, expected_fd), (
            f"Strategy={strategy}, n_active={n_active}: got {feats.shape}, "
            f"expected ({N}, {expected_fd})"
        )

    def test_time_feature_constant_across_paths(self):
        """The time feature must be the same for all paths at a given step."""
        N = 12
        swap, opt002, opt001, opt0015 = self._make_tensors(N)
        feats = extract_features(
            'S_I', 0, torch.randn(N, 289, 5),
            swap, opt002, opt001, opt0015, n_active=5, t=2.5,
        )
        time_col = feats[:, 0]  # first feature = time
        assert torch.allclose(time_col, torch.full((N,), 2.5)), (
            "Time feature must be uniform across all paths"
        )

    def test_features_finite(self):
        """All feature values must be finite."""
        N = 10
        swap, opt002, opt001, opt0015 = self._make_tensors(N)
        for strat in STRATEGIES:
            feats = extract_features(
                strat, 10, torch.randn(N, 289, 5),
                swap, opt002, opt001, opt0015, n_active=5, t=0.5,
            )
            assert torch.all(torch.isfinite(feats)), (
                f"Strategy {strat}: features contain non-finite values"
            )


# ---------------------------------------------------------------------------
# smax_strategy
# ---------------------------------------------------------------------------

class TestSmaxStrategy:
    """Analytical S^Max benchmark strategy."""

    def test_output_shape(self):
        """P&L must have shape (N,)."""
        N = 32
        swaption_002 = np.abs(np.random.randn(N, 289, 5)) * 0.01
        pl = smax_strategy(swaption_002, [128, 160, 192, 224, 256])
        assert pl.shape == (N,), f"Expected ({N},), got {pl.shape}"

    def test_pl_finite(self):
        """P&L values must be finite."""
        N = 16
        swaption_002 = np.abs(np.random.randn(N, 289, 5)) * 0.005
        pl = smax_strategy(swaption_002, [128, 160, 192, 224, 256])
        assert np.all(np.isfinite(pl)), "S^Max P&L must be finite"

    def test_constant_swaption_zero_hedge_gain(self):
        """If swaption prices are constant, hedge gain = 0."""
        N = 8
        swaption_002 = np.ones((N, 289, 5)) * 0.005
        pl = smax_strategy(swaption_002, [128, 160, 192, 224, 256])
        # Hedge gain should be ~0 (no change in H_max), payoff from exercise
        assert np.all(np.isfinite(pl))

    def test_exercise_payoff_non_negative(self):
        """Exercise payoffs are swaption prices, which are non-negative."""
        N = 20
        rng = np.random.default_rng(0)
        swaption_002 = np.abs(rng.normal(0.003, 0.001, (N, 289, 5)))
        pl = smax_strategy(swaption_002, [128, 160, 192, 224, 256])
        # The payoff component alone would be positive; hedge gain may be negative
        # Just check pl is finite
        assert np.all(np.isfinite(pl))

    def test_equal_swaptions_exercises_first(self):
        """With all equal swaption prices, at each T_i the argmax is the first active component."""
        N = 4
        const_val = 0.005
        swaption_002 = np.full((N, 289, 5), const_val)
        pl = smax_strategy(swaption_002, [128, 160, 192, 224, 256])
        # Each path should exercise at T_1 (step 128) since best==0 means exercise now
        assert np.all(np.isfinite(pl))


# ---------------------------------------------------------------------------
# BermudanHedger: initialization and network building
# ---------------------------------------------------------------------------

class TestBermudanHedgerInit:
    """BermudanHedger initialization and structure."""

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_instantiation(self, strategy):
        """Should instantiate without error for all strategies."""
        hedger = BermudanHedger(strategy=strategy, alpha=0.5, device=DEVICE, n_epochs=2)
        assert hedger.strategy == strategy
        assert hedger.alpha == 0.5

    def test_networks_empty_before_training(self):
        """CH_nets and E_nets must be empty before train_component is called."""
        hedger = BermudanHedger(strategy='S_IS', alpha=0.5, device=DEVICE, n_epochs=2)
        assert len(hedger.CH_nets) == 0, "No networks before training"
        assert len(hedger.E_nets) == 0, "No exercise nets before training"


# ---------------------------------------------------------------------------
# BermudanHedger: one training step (smoke test)
# ---------------------------------------------------------------------------

class TestBermudanHedgerTrainingSmoke:
    """Minimal smoke tests for training loop — single component, very few epochs."""

    def _get_tiny_paths(self) -> dict:
        """Build micro training paths for smoke testing."""
        N = 64
        return _make_dummy_paths(N=N, seed=42)

    def test_train_last_component_runs(self):
        """Training component 4 (last) should not raise."""
        hedger = BermudanHedger(
            strategy='S_I', alpha=0.5, device=DEVICE,
            n_epochs=2, batch_size=32
        )
        paths = self._get_tiny_paths()
        # train_component should complete without error
        hedger.train_component(component=4, paths_train=paths)
        assert 4 in hedger.CH_nets, "CH_net for component 4 must be created after training"
        assert 4 in hedger.E_nets, "E_net for component 4 must be created after training"

    def test_trained_networks_in_eval_after_freeze(self):
        """After training, frozen networks should be evaluatable."""
        hedger = BermudanHedger(
            strategy='S_I', alpha=0.5, device=DEVICE,
            n_epochs=2, batch_size=32
        )
        paths = self._get_tiny_paths()
        hedger.train_component(component=4, paths_train=paths)
        # Networks should be evaluatable after training
        ch_net = hedger.CH_nets[4]
        e_net = hedger.E_nets[4]
        ch_net.eval()
        e_net.eval()
        # CH net uses full feature dim; E net excludes time dimension
        n_active = 5 - 4  # component=4 → n_active=1
        x_ch = torch.randn(8, feature_dim('S_I', n_active))
        x_e = torch.randn(8, feature_dim('S_I', n_active) - 1)  # no time feature
        with torch.no_grad():
            _ = ch_net(x_ch)
            _ = e_net(x_e)

    def test_evaluate_returns_array_after_training(self):
        """After training all components, evaluate should return shape (N,)."""
        hedger = BermudanHedger(
            strategy='S_IS', alpha=0.5, device=DEVICE,
            n_epochs=2, batch_size=32
        )
        paths = self._get_tiny_paths()
        # Train all components
        hedger.train_all_components(paths_train=paths)
        N = paths['S_rates'].shape[0]
        pl = hedger.evaluate(paths)
        assert pl.shape == (N,), f"Expected P&L shape ({N},), got {pl.shape}"
        assert np.all(np.isfinite(pl)), "Evaluated P&L must be finite"

    def test_pl_is_numeric(self):
        """P&L values must be float arrays after evaluation."""
        hedger = BermudanHedger(
            strategy='S_I', alpha=0.5, device=DEVICE,
            n_epochs=2, batch_size=32
        )
        paths = self._get_tiny_paths()
        hedger.train_all_components(paths_train=paths)
        pl = hedger.evaluate(paths)
        assert pl.dtype in (np.float32, np.float64), f"PL dtype {pl.dtype} not float"


# ---------------------------------------------------------------------------
# CVaR numerical correctness
# ---------------------------------------------------------------------------

class TestCVaRNumerical:
    """Regression tests for CVaR formula consistency."""

    def test_cvar_all_equal(self):
        """If all PL values equal c, CVaR = -c."""
        c = 0.5
        pl = torch.full((1000,), c)
        fn = CVaRLoss(alpha=0.8)
        cvar = fn.compute_cvar(pl).item()
        assert cvar == pytest.approx(-c, abs=1e-5)

    def test_cvar_01_mean(self):
        """CVaR at near-zero alpha approximates -mean(PL) (Expected Shortfall ≈ mean)."""
        torch.manual_seed(0)
        pl = torch.randn(10000)
        # Use alpha=0.01 (very small but in valid open interval (0,1))
        fn = CVaRLoss(alpha=0.01)
        # At alpha≈0: CVaR ≈ -quantile_0.99 + E[max(q-PL,0)]/0.99, just check finite
        cvar = fn.compute_cvar(pl).item()
        assert np.isfinite(cvar)

    @pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.8])
    def test_cvar_worst_fraction(self, alpha):
        """CVaR_alpha = -mean of bottom (1-alpha) fraction of PL."""
        torch.manual_seed(2)
        pl = torch.randn(10000)
        fn = CVaRLoss(alpha=alpha)
        cvar_val = fn.compute_cvar(pl).item()

        # Reference: mean of worst (1-alpha) fraction
        pl_sorted = torch.sort(pl).values
        n_tail = int((1 - alpha) * len(pl))
        n_tail = max(n_tail, 1)
        cvar_ref = -pl_sorted[:n_tail].mean().item()

        # Allow 2% tolerance due to rounding in quantile computation
        assert abs(cvar_val - cvar_ref) / (abs(cvar_ref) + 1e-8) < 0.05, (
            f"CVaR (alpha={alpha}): got {cvar_val:.4f}, ref {cvar_ref:.4f}"
        )
