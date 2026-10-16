"""joint_inclusion against exact enumeration of all C(N, n) samples.

For CPS, the probability of sample s is prod_{k in s} w_k / sum_{z in S_n} prod_{k in z} w_k,
so the exact pi_ij is the sum of p(s) over the samples that contain both i and j.
"""

from itertools import combinations

import numpy as np
import pytest

from cps_sampling import calibrate, draw_matrix, inclusion_from_matrix, joint_inclusion


def exact_Pi_by_enumeration(w, n):
    N = len(w)
    samples = list(combinations(range(N), n))
    log_probs = np.array([np.sum(np.log(w[list(s)])) for s in samples])
    probs = np.exp(log_probs - np.max(log_probs))
    probs /= np.sum(probs)

    Pi = np.zeros((N, N))
    for prob, s in zip(probs, samples):
        Pi[np.ix_(s, s)] += prob
    return Pi


@pytest.mark.parametrize("N, n, seed", [(6, 2, 1), (6, 3, 2), (8, 3, 3), (10, 4, 4), (20, 5, 5), (50, 8, 6)])
def test_joint_inclusion_matches_enumeration(N, n, seed):
    a = np.random.default_rng(seed).uniform(0.0, 1.0, N)
    pi = np.clip(n * a / np.sum(a), 1e-10, 1 - 1e-10)
    p = calibrate(pi=pi, eps=1e-10, max_iter=500)

    # calibrate stops early once its step stalls, so the design's pi matches the target to about 1e-4
    assert np.max(np.abs(inclusion_from_matrix(q=draw_matrix(p=p, n=n)) - pi)) < 1e-3

    Pi = joint_inclusion(p=p, n=n)
    if N <= 20:
        assert np.max(np.abs(Pi - exact_Pi_by_enumeration(w=p / (1 - p), n=n))) < 1e-10
    assert np.max(np.abs(Pi - Pi.T)) < 1e-12
    # sum_{j != i} pi_ij = (n - 1) pi_i
    assert np.max(np.abs(Pi.sum(axis=1) - np.diag(Pi) - (n - 1) * np.diag(Pi))) < 1e-10
