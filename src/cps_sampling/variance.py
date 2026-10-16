"""Design-based variance estimators for the Horvitz-Thompson mean (1/N) sum_{k in s} y_k / pi_k.

These are the estimators compared by Matei & Tillé (2005) for CPS. Each returns the variance of
the mean, already divided by N^2; pass N=1 for the variance of the HT total.
deville_variance needs only first-order pi; syg_variance and ht_variance need second-order Pi.
"""

import numpy as np


def ht_variance(y, pi, Pi, s=None, N=None):
    """
    Horvitz-Thompson (expansion) variance.

    If s is None, computes the TRUE variance over the full population:
        V[That_pi] = (1/N^2) * sum_i sum_j (y_i*y_j)/(pi_i*pi_j) * (pi_ij - pi_i*pi_j)

    If s is provided, computes the ESTIMATED variance from the sample:
        Vhat = (1/N^2) * sum_{i in s} sum_{j in s} (y_i*y_j)/(pi_i*pi_j) * (pi_ij - pi_i*pi_j) / pi_ij

    Parameters
    ----------
    y  : array of ALL values (length N); only y[s] is used when s is given
    pi : first-order inclusion probabilities (length N)
    Pi : N x N second-order inclusion probabilities (anything that supports Pi[np.ix_(idx, idx)])
    s  : indices of sampled units (None for the true variance)
    N  : population size (len(y) if None)
    """
    if N is None:
        N = len(y)

    if s is None:
        idx = np.arange(N)
    else:
        idx = s

    yi = y[idx]
    pi_i = pi[idx]
    Pi_sub = Pi[np.ix_(idx, idx)]
    pi_outer = np.outer(pi_i, pi_i)
    ypi_outer = np.outer(yi / pi_i, yi / pi_i)

    if s is None:
        var = np.sum(ypi_outer * (Pi_sub - pi_outer))
    else:
        var = np.sum(ypi_outer * (Pi_sub - pi_outer) / Pi_sub)

    return var / (N**2)


def syg_variance(y, pi, Pi, s, N=None):
    """
    Sen-Yates-Grundy variance estimator (equation 16 in Matei & Tillé 2005).

    Vhat_SYG = (1/(2*N^2)) * sum_{i in s} sum_{j!=i in s}
                    (y_i/pi_i - y_j/pi_j)^2 * (pi_i*pi_j - pi_ij) / pi_ij

    Unbiased for fixed-size designs. Requires second-order inclusion probabilities.
    """
    if N is None:
        N = len(y)

    yi = y[s]
    pi_i = pi[s]
    ypi = yi / pi_i

    Pi_sub = Pi[np.ix_(s, s)]
    pi_outer = np.outer(pi_i, pi_i)

    diff_sq = (ypi[:, None] - ypi[None, :]) ** 2

    delta = (pi_outer - Pi_sub) / Pi_sub
    np.fill_diagonal(delta, 0.0)

    var = 0.5 * np.sum(diff_sq * delta)
    return var / (N**2)


def deville_variance(y, pi, s, N=None):
    """
    Deville (1993) variance estimator using only first-order inclusion probabilities.
    Works for any fixed-size design; recommended by Matei & Tillé (2005) for CPS.

    Vhat = (1/N^2) * sum_{k in s} (c_k / pi_k^2) * (y_k - y*_k)^2

    where c_k = (1 - pi_k) * n/(n-1)
    and   y*_k = pi_k * sum(c_l * y_l/pi_l) / sum(c_l)
    """
    if N is None:
        N = len(y)

    n = len(s)
    c = (1 - pi[s]) * n / (n - 1)
    y_over_pi = y[s] / pi[s]

    y_star = pi[s] * np.sum(c * y_over_pi) / np.sum(c)

    var = np.sum(c / pi[s] ** 2 * (y[s] - y_star) ** 2)
    return var / (N**2)
