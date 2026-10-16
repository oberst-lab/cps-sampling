"""The CPS design: give it inclusion probabilities pi, draw samples of exactly n = sum(pi) units.

Building the design (calibrate p, then fill the draw matrix q) is done once. Each draw then walks
through the units once, reading q, so drawing many samples costs O(N) each.
"""

import numpy as np

from cps_sampling._kernels import pi_from_q, q_from_w, s_from_q, seed_numba
from cps_sampling.calibration import calibrate
from cps_sampling.joint import joint_inclusion
from cps_sampling.variance import deville_variance, ht_variance, syg_variance


def draw_matrix(p, n):
    """The N x n draw matrix q for CPS with working probabilities p and size n.

    q[k, z] = P(include unit k | z+1 units still needed from units k, ..., N-1).
    """
    return q_from_w(p / (1 - p), n)


def inclusion_from_matrix(q):
    """The inclusion probabilities of the design that q draws from. Equal to the target pi after calibration."""
    return pi_from_q(q)


def sample_from_matrix(q, seed):
    """A 0/1 vector of length N with exactly n ones. The same (q, seed) always gives the same sample."""
    seed_numba(seed)
    return s_from_q(q)


class ConditionalPoisson:
    """
    Conditional Poisson sampling (maximum entropy sampling with fixed size) with inclusion probabilities pi.

    pi        : length-N target inclusion probabilities in [0, 1], summing to n.
                Units with pi exactly 1 are always sampled; the others are clipped to [gamma, 1 - gamma].
    n         : the sample size. Raises ValueError if sum(pi) differs from n (relative tolerance 1e-6).
    gamma     : clipping margin for the other units
    cache_dir : folder to cache the calibration in, or None for no cache

    Attributes: pi (the clipped pi), n, N, certain (mask of the pi = 1 units), and for the other units
    p (working probabilities) and q (draw matrix).
    """

    def __init__(self, pi, n, gamma=1e-10, cache_dir=None):
        pi = np.asarray(pi, dtype=np.float64)
        if pi.ndim != 1:
            raise ValueError("pi must be a 1-D array")
        if np.any(pi < 0) or np.any(pi > 1):
            raise ValueError("pi must be in [0, 1]")

        self.n = n
        self.N = len(pi)
        self.certain = pi >= 1.0
        n_certain = int(np.sum(self.certain))
        if n_certain > n:
            raise ValueError(
                f"number of certainty units ({n_certain}) exceeds n ({n}); "
                "sample your certainty units uniformly instead"
            )
        total = np.sum(pi)
        if not np.isclose(total, n, rtol=1e-6, atol=1e-6):
            raise ValueError(f"sum(pi) = {total:.6f} but n = {n}; rescale pi to sum to n")

        self.pi = np.where(self.certain, 1.0, np.clip(pi, gamma, 1 - gamma))
        self._others = np.where(~self.certain)[0]
        self._n_others = self.n - n_certain
        self.p = None
        self.q = None
        self._Pi = None
        if self._n_others == 0:
            return

        self.p = calibrate(pi=self.pi[self._others], cache_dir=cache_dir, gamma=gamma)
        self.q = draw_matrix(p=self.p, n=self._n_others)

    def sample(self, seed):
        """Sorted indices of exactly n sampled units. The same seed always gives the same sample."""
        indicator = self.certain.astype(np.float64)
        if self.q is not None:
            indicator[self._others] = sample_from_matrix(q=self.q, seed=seed)
        return np.where(indicator)[0]

    def joint_inclusion(self):
        """N x N second-order inclusion probabilities, computed on the first call. O(n N^2) time, O(N^2) memory."""
        if self._Pi is not None:
            return self._Pi

        # A certainty unit is in every sample, so pi_ij = pi_j whenever i or j is certain.
        Pi = np.minimum.outer(self.pi, self.pi)
        Pi[np.ix_(self.certain, self.certain)] = 1.0
        if self.p is not None:
            Pi[np.ix_(self._others, self._others)] = joint_inclusion(p=self.p, n=self._n_others)
        self._Pi = Pi
        return Pi

    def variance(self, y, s, method="deville", N=None):
        """
        Estimated variance of the HT mean (1/N) sum_{k in s} y_k / pi_k from the sample s.

        y      : length-N values (only y[s] is used)
        method : "deville" (needs only pi; recommended by Matei & Tillé 2005), "syg" or "ht" (both build Pi)
        N      : the divisor; defaults to len(y). Pass N=1 for the variance of the HT total.
        """
        y = np.asarray(y, dtype=np.float64)
        if method == "deville":
            return deville_variance(y=y, pi=self.pi, s=s, N=N)
        if method == "syg":
            return syg_variance(y=y, pi=self.pi, Pi=self.joint_inclusion(), s=s, N=N)
        if method == "ht":
            return ht_variance(y=y, pi=self.pi, Pi=self.joint_inclusion(), s=s, N=N)
        raise ValueError(f"unknown method {method!r}; use 'deville', 'syg' or 'ht'")
