import numpy as np
import pytest

from cps_sampling import ConditionalPoisson


def make_pi(N, n, seed):
    a = np.random.default_rng(seed).uniform(0.05, 1.0, N)
    return n * a / np.sum(a)


def test_fixed_size_and_reproducible():
    design = ConditionalPoisson(pi=make_pi(N=200, n=20, seed=0), n=20)
    for seed in range(50):
        s = design.sample(seed=seed)
        assert len(s) == 20
        assert len(np.unique(s)) == 20
        assert np.array_equal(s, design.sample(seed=seed))
    assert not np.array_equal(design.sample(seed=0), design.sample(seed=1))


def test_empirical_inclusion_matches_pi():
    pi = make_pi(N=30, n=6, seed=1)
    design = ConditionalPoisson(pi=pi, n=6)
    counts = np.zeros(30)
    n_draws = 20000
    for seed in range(n_draws):
        counts[design.sample(seed=seed)] += 1
    # 4 binomial standard errors
    assert np.all(np.abs(counts / n_draws - pi) < 4 * np.sqrt(pi * (1 - pi) / n_draws) + 1e-3)


def test_certainty_units_always_sampled():
    pi = make_pi(N=40, n=6, seed=2)
    pi[:3] = 1.0
    pi[3:] *= 5 / pi[3:].sum()  # the other units carry n - 3 = 5
    design = ConditionalPoisson(pi=pi, n=8)
    for seed in range(100):
        s = design.sample(seed=seed)
        assert len(s) == 8
        assert {0, 1, 2} <= set(s)

    Pi = design.joint_inclusion()
    assert np.allclose(np.diag(Pi), design.pi, atol=1e-3)
    assert np.allclose(Pi[0, 5:], design.pi[5:])


def test_only_certainty_units():
    design = ConditionalPoisson(pi=np.array([1.0, 1.0, 0.0, 0.0]), n=2)
    assert np.array_equal(design.sample(seed=0), [0, 1])


def test_pi_out_of_range_raises():
    with pytest.raises(ValueError):
        ConditionalPoisson(pi=np.array([0.5, 1.5]), n=2)


def test_sum_not_n_raises():
    with pytest.raises(ValueError, match="sum"):
        ConditionalPoisson(pi=make_pi(N=50, n=5, seed=0), n=6)


def test_too_many_certainty_units_raises():
    with pytest.raises(ValueError, match="certainty units"):
        ConditionalPoisson(pi=np.array([1.0, 1.0, 1.0, 0.0]), n=2)
