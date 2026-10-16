"""Conditional Poisson sampling: give it inclusion probabilities pi, get samples of exactly n units."""

from cps_sampling.calibration import calibrate
from cps_sampling.design import ConditionalPoisson, draw_matrix, inclusion_from_matrix, sample_from_matrix
from cps_sampling.joint import joint_inclusion
from cps_sampling.variance import deville_variance, ht_variance, syg_variance

__version__ = "0.1.0"

__all__ = [
    "ConditionalPoisson",
    "calibrate",
    "draw_matrix",
    "inclusion_from_matrix",
    "sample_from_matrix",
    "joint_inclusion",
    "deville_variance",
    "syg_variance",
    "ht_variance",
]
