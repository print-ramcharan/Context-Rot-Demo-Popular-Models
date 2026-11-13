
from evaluation.metrics.statistical_tests import holm_bonferroni


def test_holm_bonferroni_adjustment():
    p_values = [0.01, 0.04, 0.03, 0.002]

    adjusted = holm_bonferroni(p_values)

    expected = [0.03, 0.06, 0.06, 0.008]

    for actual, target in zip(adjusted, expected):
        assert abs(actual - target) < 1e-10


def test_holm_preserves_original_order():
    p_values = [0.04, 0.001, 0.02]

    adjusted = holm_bonferroni(p_values)

    assert adjusted == [0.04, 0.003, 0.04]


def test_holm_handles_none():
    p_values = [0.01, None, 0.04]

    adjusted = holm_bonferroni(p_values)

    assert adjusted[1] is None
    assert adjusted[0] == 0.02
    assert adjusted[2] == 0.04
