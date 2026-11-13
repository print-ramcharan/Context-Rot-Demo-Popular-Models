import pytest

from evaluation.metrics.statistical_tests import (
    holm_bonferroni,
    paired_bootstrap_ci,
    wilcoxon_signed_rank_test,
)


def test_wilcoxon_returns_none_for_single_question():
    assert wilcoxon_signed_rank_test([0.0], [1.0]) is None


def test_wilcoxon_handles_all_zero_differences_without_warning():
    assert wilcoxon_signed_rank_test([0.0, 1.0, 0.5], [0.0, 1.0, 0.5]) == (0.0, 1.0)


def test_wilcoxon_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        wilcoxon_signed_rank_test([0.0, float("nan")], [1.0, 1.0])


def test_bootstrap_is_deterministic_and_finite():
    first = paired_bootstrap_ci([0.1, 0.2, -0.1], n_resamples=500, seed=7)
    second = paired_bootstrap_ci([0.1, 0.2, -0.1], n_resamples=500, seed=7)
    assert first == second
    assert first[0] <= first[1]


def test_holm_preserves_none_and_original_order():
    assert holm_bonferroni([0.01, None, 0.04]) == [0.02, None, 0.04]
