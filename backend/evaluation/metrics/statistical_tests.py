"""Statistical helpers for paired benchmark comparisons.

The caller must provide one paired observation per benchmark question.
One-question samples are descriptive only and are not tested for significance.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

RANDOM_SEED = 42


def paired_bootstrap_ci(
    differences: Sequence[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference."""
    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0:
        raise ValueError("At least one paired difference is required")
    if not np.all(np.isfinite(diffs)):
        raise ValueError("Paired differences must all be finite")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    n = diffs.size
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        means[i] = np.mean(rng.choice(diffs, size=n, replace=True))
    alpha = 1.0 - confidence
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def wilcoxon_signed_rank_test(
    before: Sequence[float],
    after: Sequence[float],
) -> tuple[float, float] | None:
    """Paired two-sided Wilcoxon test; return None when n<2.

    All-zero differences return (0, 1). Non-finite or unpaired inputs fail
    loudly rather than silently producing misleading p-values.
    """
    before_arr = np.asarray(before, dtype=float)
    after_arr = np.asarray(after, dtype=float)
    if before_arr.size != after_arr.size:
        raise ValueError("Paired samples must have equal lengths")
    if before_arr.size < 2:
        return None
    if not np.all(np.isfinite(before_arr)) or not np.all(np.isfinite(after_arr)):
        raise ValueError("Paired samples must all be finite")

    differences = after_arr - before_arr
    if np.all(differences == 0):
        return (0.0, 1.0)
    result = stats.wilcoxon(
        differences,
        zero_method="pratt",
        alternative="two-sided",
        correction=False,
        method="auto",
    )
    return (float(result.statistic), float(result.pvalue))


def holm_bonferroni(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    values = list(p_values)
    adjusted: list[float | None] = [None] * len(values)
    valid = []
    for index, value in enumerate(values):
        if value is None:
            continue
        p_value = float(value)
        if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError("p-values must be finite and between 0 and 1")
        valid.append((index, p_value))

    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        corrected = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max
    return adjusted
