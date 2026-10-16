"""Numba kernels for conditional Poisson sampling, moved verbatim from p-erik-cps-code/core/cps.py.

We work with log-weights throughout the elementary symmetric polynomial computation,
since those polynomials overflow or underflow for extreme weights.
The q values themselves are bounded in [0, 1], so we only exponentiate at the end.
"""

import numpy as np
from numba import njit, prange


@njit
def cumulative_logsumexp_reverse(log_w):
    """Compute logsumexp(log_w[i:]) for all i in O(N) time."""
    N = len(log_w)
    result = np.empty(N)
    result[N - 1] = log_w[N - 1]
    for i in range(N - 2, -1, -1):
        a, b = log_w[i], result[i + 1]
        max_val = max(a, b)
        if max_val == -np.inf:
            result[i] = -np.inf
        else:
            result[i] = max_val + np.log(np.exp(a - max_val) + np.exp(b - max_val))
    return result


@njit
def cumulative_logsum_reverse(log_w):
    """Compute sum(log_w[i:]) for all i in O(N) time."""
    N = len(log_w)
    result = np.empty(N)
    result[N - 1] = log_w[N - 1]
    for i in range(N - 2, -1, -1):
        result[i] = log_w[i] + result[i + 1]
    return result


@njit
def logsumexp_pair(a, b):
    """Stable logsumexp for two values."""
    max_val = max(a, b)
    if max_val == -np.inf:
        return -np.inf
    return max_val + np.log(np.exp(a - max_val) + np.exp(b - max_val))


@njit(parallel=True)
def q_from_w(w, n):
    """
    Compute the q matrix from weights w, working in log-space for stability.

    q[i, z] = P(include unit i | z+1 units still needed from units i, ..., N-1).

    The elementary symmetric polynomial e_k(w[i:]) satisfies:
        e_k(w[i:]) = w[i] * e_{k-1}(w[i+1:]) + e_k(w[i+1:])

    We store log(e_k) in log_expa to avoid overflow.
    """
    N = len(w)

    # Work with log-weights
    log_w = np.empty(N)
    for i in prange(N):
        log_w[i] = np.log(max(w[i], 1e-300))

    # log_expa[i, z] = log(e_{z+1}(w[i:]))
    log_expa = np.full((N, n), -np.inf)

    # Base case: e_1(w[i:]) = sum of w[i:] - O(N) using cumulative computation
    log_expa[:, 0] = cumulative_logsumexp_reverse(log_w)

    # Precompute cumulative log sums for the boundary condition
    cumsum_log = cumulative_logsum_reverse(log_w)

    # Recurrence: e_k(w[i:]) = w[i] * e_{k-1}(w[i+1:]) + e_k(w[i+1:])
    # The recurrence must be sequential in i (backwards)
    for i in range(N - 2, -1, -1):
        remaining = N - i
        max_col = min(n, remaining)

        for z in range(1, max_col):
            term1 = log_w[i] + log_expa[i + 1, z - 1]
            term2 = log_expa[i + 1, z]
            log_expa[i, z] = logsumexp_pair(term1, term2)

        # Boundary: when remaining == z+1, e_{z+1}(w[i:]) = prod(w[i:])
        if remaining <= n:
            col = remaining - 1
            log_expa[i, col] = cumsum_log[i]

    # Now compute q matrix
    # q[i, z] = P(include item i | need z+1 more items from i onwards)
    #         = w[i] * e_z(w[i+1:]) / e_{z+1}(w[i:])
    q = np.zeros((N, n))

    # Column 0: q[i, 0] = w[i] / e_1(w[i:]) = w[i] / sum(w[i:])
    for i in prange(N):
        log_q = log_w[i] - log_expa[i, 0]
        q[i, 0] = np.exp(min(max(log_q, -700), 0))

    # Other columns
    for i in prange(N - 1):
        for z in range(1, n):
            log_num = log_w[i] + log_expa[i + 1, z - 1]
            log_den = log_expa[i, z]
            log_q = log_num - log_den

            if np.isnan(log_q) or (np.isinf(log_q) and log_q < 0):
                q[i, z] = 0.0
            else:
                q[i, z] = np.exp(min(max(log_q, -700), 0))

    # Boundary condition: if we have exactly k items left and need k,
    # we must include all of them -> q = 1
    for i in prange(N):
        remaining = N - i
        if remaining <= n:
            q[i, remaining - 1] = 1.0

    # Safety: clip q to [0, 1]
    for i in prange(N):
        for z in range(n):
            q[i, z] = min(max(q[i, z], 0.0), 1.0)

    return q


@njit(parallel=True)
def pi_from_q(q):
    """Compute inclusion probabilities from the q matrix."""
    N, n_target = q.shape
    P = np.zeros((N + 1, n_target + 1))
    P[0, n_target] = 1.0

    # Sequential - has dependencies
    for k in range(N):
        for j in range(1, n_target + 1):
            q_kj = q[k, j - 1]
            P[k + 1, j - 1] += P[k, j] * q_kj
            P[k + 1, j] += P[k, j] * (1 - q_kj)
        P[k + 1, 0] += P[k, 0]

    # Parallel final computation
    pik = np.zeros(N)
    for i in prange(N):
        s = 0.0
        for j in range(1, n_target + 1):
            s += P[i, j] * q[i, j - 1]
        pik[i] = s

    return pik


@njit
def s_from_q(q):
    """Draw a sample (a 0/1 vector) from the q matrix with Numba's RNG. Seed that RNG first with seed_numba."""
    N, n = q.shape
    s = np.zeros(N)
    remaining = n

    for k in range(N):
        if remaining > 0:
            if np.random.rand() < q[k, remaining - 1]:
                s[k] = 1
                remaining -= 1
        else:
            break
    return s


@njit
def seed_numba(seed):
    """Seed Numba's RNG. np.random.seed called outside @njit seeds only NumPy's RNG, which Numba does not use."""
    np.random.seed(seed)
