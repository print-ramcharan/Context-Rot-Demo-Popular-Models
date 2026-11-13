"""Statistical significance helpers.

Provides paired bootstrap confidence intervals and a paired Wilcoxon
signed-rank test for per-query retrieval metrics.
"""

import numpy as np
from scipy import stats
from typing import Sequence, Tuple

RANDOM_SEED = 42


def paired_bootstrap_ci(
    differences: Sequence[float], n_resamples: int = 10000, confidence: float = 0.95, seed: int = RANDOM_SEED
) -> Tuple[float, float]:
    """Paired bootstrap confidence interval for the mean of paired differences."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0:
        return (0.0, 0.0)
    means = np.empty(n_resamples, dtype=float)
    n = diffs.size
    for i in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        means[i] = np.mean(sample)
    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def wilcoxon_signed_rank_test(before: Sequence[float], after: Sequence[float]):
    """Paired Wilcoxon signed-rank test on per-query metric differences."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.size != after.size or before.size == 0:
        return None
    # Use a small constant to avoid zero differences being dropped by scipy.
    try:
        stat, pvalue = stats.wilcoxon(before, after, zero_method="wilcox", correction=False)
    except ValueError:
        # All differences are zero -> no evidence of a shift.
        return (0.0, 1.0)
    return (float(stat), float(pvalue))
