"""Phase 1 gate: the platform's error rates are themselves under test.

Monte Carlo verification that (a) the fixed-horizon test holds its
false-positive rate, (b) designs sized by the power calculator achieve their
promised power, (c) peeking with a naive threshold inflates FPR (the disease
is real), and (d) OBF sequential boundaries cure it. Fixed seeds throughout —
failures are regressions, not noise.
"""

from __future__ import annotations

import itertools
from statistics import NormalDist

import numpy as np

from growth_lab.experiments import (
    SequentialDesign,
    cuped_adjust,
    cuped_theta,
    minimum_detectable_effect,
    obf_boundaries,
    required_n_per_arm,
    two_proportion_ztest,
    welch_test,
)

_NORMAL = NormalDist()


def _fpr_two_proportion(n_sims: int, n_per_arm: int, p: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    x_c = rng.binomial(n_per_arm, p, size=n_sims)
    x_t = rng.binomial(n_per_arm, p, size=n_sims)
    rejections = sum(
        two_proportion_ztest(int(c), n_per_arm, int(t), n_per_arm).significant
        for c, t in zip(x_c, x_t, strict=True)
    )
    return rejections / n_sims


def test_aa_false_positive_rate_calibrated() -> None:
    """A/A: rejection rate must sit within Monte Carlo error of alpha=0.05."""
    fpr = _fpr_two_proportion(n_sims=4000, n_per_arm=20_000, p=0.05, seed=7)
    se = (0.05 * 0.95 / 4000) ** 0.5  # ~0.0034
    assert abs(fpr - 0.05) < 4 * se, f"A/A FPR {fpr:.4f} is not calibrated"


def test_power_promise_is_kept() -> None:
    """Designs sized for 80% power must reject ~80% of the time at the MDE."""
    p_base, mde = 0.05, 0.10
    n = required_n_per_arm(p_base, mde, alpha=0.05, power=0.8)
    rng = np.random.default_rng(11)
    n_sims = 2000
    x_c = rng.binomial(n, p_base, size=n_sims)
    x_t = rng.binomial(n, p_base * (1 + mde), size=n_sims)
    rejections = sum(
        two_proportion_ztest(int(c), n, int(t), n).significant
        for c, t in zip(x_c, x_t, strict=True)
    )
    realized_power = rejections / n_sims
    assert abs(realized_power - 0.8) < 0.04, (
        f"promised power 0.80, realized {realized_power:.3f} at n={n}"
    )


def test_mde_inverts_sample_size() -> None:
    n = required_n_per_arm(0.05, 0.10)
    recovered = minimum_detectable_effect(n, 0.05)
    assert abs(recovered - 0.10) < 0.002


def test_peeking_inflates_false_positives() -> None:
    """The disease: 5 looks at naive z=1.96 nearly triples the error rate."""
    rng = np.random.default_rng(23)
    n_sims, n_looks = 4000, 5
    increments = rng.standard_normal((n_sims, n_looks))
    z = increments.cumsum(axis=1) / np.sqrt(np.arange(1, n_looks + 1))
    crit = _NORMAL.inv_cdf(0.975)
    fpr = float((np.abs(z) > crit).any(axis=1).mean())
    assert fpr > 0.10, f"peeking inflation not reproduced: {fpr:.4f}"


def test_obf_sequential_controls_false_positives() -> None:
    """The cure: OBF boundaries hold overall FPR at alpha across all looks."""
    design = SequentialDesign.create(n_looks=5, alpha=0.05)
    rng = np.random.default_rng(29)
    n_sims = 4000
    increments = rng.standard_normal((n_sims, design.n_looks))
    z = increments.cumsum(axis=1) / np.sqrt(np.arange(1, design.n_looks + 1))
    crossed = (np.abs(z) > np.asarray(design.boundaries)).any(axis=1)
    fpr = float(crossed.mean())
    se = (0.05 * 0.95 / n_sims) ** 0.5
    assert abs(fpr - 0.05) < 4 * se + 0.005, f"sequential FPR {fpr:.4f} not calibrated"


def test_obf_boundaries_match_literature() -> None:
    """K=5, alpha=0.05 Lan-DeMets OBF boundaries (e.g. 2.031 at the final look)."""
    reference = (4.877, 3.357, 2.680, 2.290, 2.031)
    computed = obf_boundaries(5, 0.05)
    for ref, got in zip(reference[2:], computed[2:], strict=True):
        assert abs(got - ref) / ref < 0.02, f"boundary {got:.3f} vs literature {ref:.3f}"
    # early-look boundaries are extreme-tail; allow looser agreement
    for ref, got in zip(reference[:2], computed[:2], strict=True):
        assert abs(got - ref) / ref < 0.06, f"boundary {got:.3f} vs literature {ref:.3f}"


def test_obf_boundaries_decrease() -> None:
    b = obf_boundaries(5, 0.05)
    assert all(earlier > later for earlier, later in itertools.pairwise(b))
    assert b[-1] > _NORMAL.inv_cdf(0.975)  # final look still stricter than fixed test


def test_cuped_reduces_variance_and_stays_calibrated() -> None:
    """CUPED must cut variance by ~rho^2 and keep the A/A FPR at alpha."""
    rng = np.random.default_rng(31)
    n_sims, n = 800, 2000
    rho2_target = 0.5
    rejections_plain = 0
    rejections_cuped = 0
    var_ratios = []
    for _ in range(n_sims):
        pre = rng.standard_normal(2 * n)
        noise = rng.standard_normal(2 * n)
        y = np.sqrt(rho2_target) * pre + np.sqrt(1 - rho2_target) * noise
        theta = cuped_theta(y, pre)
        y_adj = cuped_adjust(y, pre, theta)
        var_ratios.append(float(y_adj.var() / y.var()))
        rejections_plain += welch_test(y[:n], y[n:]).significant
        rejections_cuped += welch_test(y_adj[:n], y_adj[n:]).significant
    assert abs(np.mean(var_ratios) - (1 - rho2_target)) < 0.02
    for fpr in (rejections_plain / n_sims, rejections_cuped / n_sims):
        assert abs(fpr - 0.05) < 0.03, f"FPR {fpr:.4f} not calibrated"
