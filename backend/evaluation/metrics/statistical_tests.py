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

def holm_bonferroni(p_values: Sequence[float]):
    """Return Holm-Bonferroni adjusted p-values in original order."""
    values = list(p_values)
    adjusted = [None] * len(values)

    valid = [
        (i, float(p))
        for i, p in enumerate(values)
        if p is not None
    ]

    if not valid:
        return adjusted

    if any(not 0.0 <= p <= 1.0 for _, p in valid):
        raise ValueError("p-values must be between 0 and 1")

    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(ordered):
        corrected = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max

    return adjusted