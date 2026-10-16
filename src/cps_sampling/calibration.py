"""Calibration: find the working probabilities p whose CPS design has the target inclusion probabilities pi.

CPS with weights w = p / (1 - p) does not have inclusion probabilities p. We find p by the
fixed-point iteration of Chen, Dempster and Liu (1994): p <- p + pi - pi(p).

This is the slow step. For a given (pi, n) the result is deterministic, so it can be cached on disk:
pass cache_dir, and files are written there as {sha256 of rounded pi}_{n}.npy.
"""

import hashlib
import logging
from pathlib import Path

import numpy as np

from cps_sampling._kernels import pi_from_q, q_from_w

logger = logging.getLogger(__name__)

_CACHE_ROUND_DECIMALS = 10


def _cache_key(pik, n):
    """Deterministic hash of (rounded pi vector, n)."""
    rounded = np.round(pik, decimals=_CACHE_ROUND_DECIMALS)
    h = hashlib.sha256(rounded.tobytes()).hexdigest()[:16]
    return f"{h}_{n}"


def calibrate(pi, cache_dir=None, gamma=1e-10, eps=0.01, max_iter=100, max_iter_no_change=5):
    """
    Working probabilities p such that CPS with weights w = p / (1 - p) has inclusion probabilities pi.

    pi         : target inclusion probabilities, clipped to [gamma, 1 - gamma]; n = round(sum(pi))
    cache_dir  : folder for the disk cache, or None for no cache
    eps        : stop when the L1 change of p in one iteration is below eps
    max_iter   : at most this many iterations
    max_iter_no_change : stop when the change has moved by less than 1e-2 over this many iterations
    """
    pik = np.clip(pi, gamma, 1 - gamma)
    n = int(np.round(np.sum(pik)))

    key = _cache_key(pik, n)
    cache_path = Path(cache_dir) / f"{key}.npy" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        logger.debug("calibrate: cache hit  (N=%d, n=%d, key=%s)", len(pik), n, key)
        return np.load(cache_path)

    logger.debug("calibrate: cache miss (N=%d, n=%d, key=%s)", len(pik), n, key)
    pikt = pik.copy()
    arr = 1.0
    arr_history = None

    for i in range(max_iter):
        # Early stopping checks
        if i >= max_iter_no_change and arr_history is not None:
            if np.abs(arr_history - arr) < 1e-2:
                logger.debug("calibrate: no change in %d iterations", max_iter_no_change)
                break
        if i % max_iter_no_change == 0:
            arr_history = arr
        if arr < eps:
            break
        if np.any(np.isnan(pikt)):
            logger.warning("calibrate: NaNs in p, stopping")
            break

        # Clip pikt to valid range before computing weights
        pikt = np.clip(pikt, gamma, 1 - gamma)

        w = pikt / (1 - pikt)
        q = q_from_w(w, n)
        pik_from_q = pi_from_q(q)

        # Iterative update
        pikt1 = pikt + pik - pik_from_q

        # Clip the update to stay in valid range
        pikt1 = np.clip(pikt1, gamma, 1 - gamma)

        arr = np.sum(np.abs(pikt - pikt1))
        pikt = pikt1

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, pikt)

    return pikt
