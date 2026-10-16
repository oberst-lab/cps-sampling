"""The variance estimators under CPS, against the true HT variance computed with exact Pi (small N)."""

import numpy as np

from cps_sampling import ConditionalPoisson, ht_variance


def make_design_and_y(N, n, seed):
    rng = np.random.default_rng(seed)
    y = rng.uniform(1, 10, N)
    props = rng.uniform(0.1, 1.0, N)  # independent of y
    return ConditionalPoisson(pi=n * props / props.sum(), n=n), y


def test_estimators_unbiased_under_cps():
    design, y = make_design_and_y(N=12, n=4, seed=42)
    true_var = ht_variance(y=y, pi=design.pi, Pi=design.joint_inclusion(), s=None, N=12)

    samples = [design.sample(seed=seed) for seed in range(5000)]
    estimates = {method: np.mean([design.variance(y=y, s=s, method=method) for s in samples]) for method in ["ht", "syg", "deville"]}

    assert abs(estimates["ht"] / true_var - 1) < 0.10
    assert abs(estimates["syg"] / true_var - 1) < 0.10
    assert abs(estimates["deville"] / true_var - 1) < 0.15  # Deville is only approximately unbiased


def test_true_variance_matches_enumeration():
    """ht_variance(s=None) equals the variance of the HT mean over every possible sample."""
    from itertools import combinations

    design, y = make_design_and_y(N=8, n=3, seed=0)
    w = design.p / (1 - design.p)
    samples = list(combinations(range(8), 3))
    probs = np.array([np.prod(w[list(s)]) for s in samples])
    probs /= probs.sum()
    estimates = np.array([np.sum(y[list(s)] / design.pi[list(s)]) / 8 for s in samples])
    brute_var = np.sum(probs * (estimates - np.sum(probs * estimates)) ** 2)

    true_var = ht_variance(y=y, pi=design.pi, Pi=design.joint_inclusion(), s=None, N=8)
    # design.pi is the calibration target; the design's actual pi differs from it by the calibration tolerance
    assert abs(true_var / brute_var - 1) < 1e-3


def test_deville_nonnegative():
    design, y = make_design_and_y(N=30, n=5, seed=0)
    for seed in range(2000):
        assert design.variance(y=y, s=design.sample(seed=seed), method="deville") >= 0


def test_constant_ratio_gives_zero_variance():
    """If y is proportional to pi, the HT mean is the same for every fixed-size sample."""
    design, _ = make_design_and_y(N=50, n=8, seed=3)
    y = 3.0 * design.pi
    for seed in range(100):
        s = design.sample(seed=seed)
        assert abs(design.variance(y=y, s=s, method="deville")) < 1e-20
        assert abs(design.variance(y=y, s=s, method="syg")) < 1e-20
