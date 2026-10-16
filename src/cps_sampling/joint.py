"""Second-order inclusion probabilities pi_ij of a CPS design."""

import numpy as np


def joint_inclusion(p, n):
    """
    N x N matrix of second-order inclusion probabilities for CPS with working probabilities p and size n.

    Uses Results 1 and 2 from Matei & Tillé (2005).
    Diagonal Pi[i,i] = pi_i; off-diagonal Pi[i,j] = P(i in s, j in s).
    O(n N^2) time and O(N^2) memory.
    """
    p = np.asarray(p, dtype=np.float64)
    w = p / (1 - p)
    N = len(p)

    phi_prev = np.zeros(N)
    phi_curr = w / np.sum(w)

    Pi_two_back = np.zeros((N, N))
    Pi_one_back = np.zeros((N, N))

    for m in range(2, n + 1):
        diag_numer = w * (1 - phi_curr)
        phi_new = m * diag_numer / np.sum(diag_numer)

        bracket = 1.0 - phi_prev[:, None] - phi_prev[None, :] + Pi_two_back
        offdiag_numer = (w[:, None] * w[None, :]) * bracket
        np.fill_diagonal(offdiag_numer, 0.0)
        Pi_new = m * (m - 1) * offdiag_numer / np.sum(offdiag_numer)
        np.fill_diagonal(Pi_new, phi_new)

        phi_prev = phi_curr
        phi_curr = phi_new
        Pi_two_back = Pi_one_back
        Pi_one_back = Pi_new.copy()

    if n >= 2:
        return Pi_one_back
    else:
        Pi = np.zeros((N, N))
        np.fill_diagonal(Pi, phi_curr)
        return Pi
