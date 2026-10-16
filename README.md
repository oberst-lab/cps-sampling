# cps-sampling

Conditional Poisson Sampling (CPS) in Python. 
You input inclusion probabilities π, where π_i = P(s_i=1) i.e. the probability of including unit i in the sample, and it gives you a sample of exactly n = Σπ units in which unit i appears with probability π_i.
This is the package released with the paper Fixed Size Statistical Inference (Skalnes and Oberst, 2026)

## Install

```bash
pip install git+https://github.com/oberst-lab/cps-sampling.git
```

## Usage

```python
import numpy as np
from cps_sampling import ConditionalPoisson

rng = np.random.default_rng(0)
uncertainty = rng.uniform(size=10_000)
pi = 500 * uncertainty / uncertainty.sum()      # inclusion probabilities summing to n = 500

design = ConditionalPoisson(pi=pi, n=500)        # slow step, done once: about 1 s for N = 25,000, n = 1,000
s = design.sample(seed=0)                        # indices of exactly 500 units; under 1 ms each

y = rng.normal(size=10_000)                      # only y[s] is used below
mean = np.sum(y[s] / pi[s]) / len(pi)            # Horvitz-Thompson mean
var = design.variance(y=y, s=s)                  # Deville's estimator, needs only pi
```

- **Sample size.** You pass n, and `ConditionalPoisson` raises an error unless sum(π) = n.
- **Certainty units.** Units with π_i = 1 are in every sample. The other units are clipped to [γ, 1 − γ], with γ = 1e-10 by default.
- **Caching.** Building the design is the slow step. `ConditionalPoisson(pi=pi, n=n, cache_dir="cache")` saves it to disk, keyed by π, so a rerun with the same π is instant.
- **Second-order probabilities.** `design.joint_inclusion()` returns the N × N matrix of π_ij (Matei and Tillé, 2005). It takes O(n N²) time, so use it only for moderate N.
- **Variance.** `design.variance(y=y, s=s, method=...)` estimates the variance of the HT mean (divided by N²; pass `N=1` for the total). `method` is `"deville"` (the default, recommended by Matei and Tillé), `"syg"` (Sen–Yates–Grundy) or `"ht"`; the last two build π_ij.

## How it works

1. **Calibration.** CPS with weights w_i = p_i / (1 − p_i) does not have inclusion probabilities p, so `calibrate` solves for the working probabilities p that give π, by the fixed point of Chen, Dempster and Liu (1994).
2. **Draw matrix.** `draw_matrix` fills q[k, z] = P(unit k is drawn | z + 1 units are still needed from units k, …, N − 1) by a recursion over elementary symmetric polynomials, in log space for stability.
3. **Drawing.** `sample_from_matrix` walks through the units once and includes unit k with probability q[k, units still needed]. The matrix is built once and reused for every draw.

## Citation

If you use this package, please cite:

```bibtex
@inproceedings{skalnes2026fixed,
  title     = {Fixed-Size Active Statistical Inference},
  author    = {Skalnes, Erik and Oberst, Michael},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## References

- Chen, Dempster and Liu (1994). Weighted finite population sampling to maximize entropy. *Biometrika* 81(3).
- Matei and Tillé (2005). Evaluation of variance approximations and estimators in maximum entropy sampling with unequal probability and fixed sample size. *Journal of Official Statistics* 21(4).
- Deville (1993). Estimation de la variance pour les enquêtes en deux phases. Manuscript, INSEE.
