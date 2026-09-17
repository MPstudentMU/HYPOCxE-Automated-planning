"""Tests for engine/stats.py (McNemar's test)."""
from __future__ import annotations

import pandas as pd
import pytest

from engine import stats as S


# --------------------------------------------------------------------------- #
# mcnemar_test — hand-computed expected values
# --------------------------------------------------------------------------- #


def test_no_discordant_pairs():
    result = S.mcnemar_test(0, 0)
    assert result.n_discordant == 0
    assert result.p_value == 1.0
    assert result.statistic is None
    assert result.method == "no discordant pairs"


def test_exact_method_below_threshold_hand_computed():
    """b=1, c=9 (n=10 < 25 -> exact). Two-sided exact binomial p-value for
    min(1,9)=1 success in 10 trials at p=0.5, by the binomial distribution's
    symmetry: 2 * [P(X<=1)] = 2 * [C(10,0)+C(10,1)] / 2^10
             = 2 * (1 + 10) / 1024 = 22/1024 = 0.021484375."""
    result = S.mcnemar_test(1, 9)
    assert result.method == "exact"
    assert result.n_discordant == 10
    assert result.statistic is None
    assert result.p_value == pytest.approx(22 / 1024, abs=1e-9)


def test_exact_method_is_symmetric_in_b_and_c():
    assert S.mcnemar_test(1, 9).p_value == pytest.approx(S.mcnemar_test(9, 1).p_value)


def test_asymptotic_method_above_threshold_hand_computed():
    """b=21, c=9 (n=30 >= 25 -> continuity-corrected chi-square).
    statistic = (|21-9|-1)^2 / 30 = 11^2/30 = 121/30 = 4.0333...
    For df=1, chi2=3.841 -> p=0.05 exactly, and 4.033 > 3.841, so p should
    land just under 0.05 (chi2.sf is cross-checked against that known
    reference point, not against itself)."""
    result = S.mcnemar_test(21, 9)
    assert result.method == "asymptotic (continuity-corrected)"
    assert result.n_discordant == 30
    assert result.statistic == pytest.approx(121 / 30, abs=1e-9)
    assert 0.03 < result.p_value < 0.05


def test_asymptotic_statistic_symmetric_in_b_and_c():
    assert S.mcnemar_test(21, 9).statistic == pytest.approx(S.mcnemar_test(9, 21).statistic)


def test_n_pairs_defaults_to_b_plus_c():
    result = S.mcnemar_test(3, 4)
    assert result.n_pairs == 7


def test_n_pairs_override_reports_concordant_pairs_too():
    result = S.mcnemar_test(3, 4, n_pairs=20)
    assert result.n_pairs == 20
    assert result.n_discordant == 7


def test_exact_threshold_is_configurable():
    result_default = S.mcnemar_test(10, 10)  # n=20, exact by default
    assert result_default.method == "exact"
    result_lowered = S.mcnemar_test(10, 10, exact_threshold=5)
    assert result_lowered.method == "asymptotic (continuity-corrected)"


# --------------------------------------------------------------------------- #
# mcnemar_manual_vs_auto
# --------------------------------------------------------------------------- #


def _row(pt_no, plan_type, goal_key, status):
    return dict(pt_no=pt_no, plan_type=plan_type, goal_key=goal_key, status=status)


def test_mcnemar_manual_vs_auto_hand_checked():
    """Manual vs Auto, paired by (pt_no, goal_key):
      g1: Manual PASS, Auto PASS  -> concordant
      g2: Manual PASS, Auto FAIL  -> b
      g3: Manual FAIL, Auto PASS  -> c
      g4: Manual FAIL, Auto FAIL  -> concordant
      g5: Manual PASS, Auto FAIL  -> b
    -> b=2, c=1."""
    df = pd.DataFrame([
        _row("Pt1", "Manual", "g1", "PASS"), _row("Pt1", "Auto", "g1", "PASS"),
        _row("Pt1", "Manual", "g2", "PASS"), _row("Pt1", "Auto", "g2", "FAIL"),
        _row("Pt1", "Manual", "g3", "FAIL"), _row("Pt1", "Auto", "g3", "PASS"),
        _row("Pt1", "Manual", "g4", "FAIL"), _row("Pt1", "Auto", "g4", "FAIL"),
        _row("Pt1", "Manual", "g5", "PASS"), _row("Pt1", "Auto", "g5", "FAIL"),
    ])
    result = S.mcnemar_manual_vs_auto(df)
    assert result is not None
    assert result.b == 2
    assert result.c == 1
    assert result.n_pairs == 5


def test_mcnemar_manual_vs_auto_pairs_across_patients():
    df = pd.DataFrame([
        _row("Pt1", "Manual", "g1", "PASS"), _row("Pt1", "Auto", "g1", "FAIL"),
        _row("Pt2", "Manual", "g1", "FAIL"), _row("Pt2", "Auto", "g1", "PASS"),
    ])
    result = S.mcnemar_manual_vs_auto(df)
    assert result.b == 1
    assert result.c == 1
    assert result.n_pairs == 2


def test_mcnemar_manual_vs_auto_ignores_unresolved_status():
    df = pd.DataFrame([
        _row("Pt1", "Manual", "g1", "PASS"), _row("Pt1", "Auto", "g1", None),
        _row("Pt1", "Manual", "g2", "PASS"), _row("Pt1", "Auto", "g2", "PASS"),
    ])
    result = S.mcnemar_manual_vs_auto(df)
    assert result.n_pairs == 1  # g1 dropped (Auto status unresolved)


def test_mcnemar_manual_vs_auto_none_when_nothing_pairs():
    df = pd.DataFrame([
        _row("Pt1", "Manual", "g1", "PASS"),
        _row("Pt1", "Auto", "g2", "PASS"),  # different goal_key -> no pair
    ])
    assert S.mcnemar_manual_vs_auto(df) is None


def test_mcnemar_manual_vs_auto_empty_df():
    df = pd.DataFrame(columns=["pt_no", "plan_type", "goal_key", "status"])
    assert S.mcnemar_manual_vs_auto(df) is None


# --------------------------------------------------------------------------- #
# correlation_pass_rate_vs_time — hand-computed via perfectly (anti)linear data
# --------------------------------------------------------------------------- #


def test_correlation_perfect_positive_linear_relationship():
    """y = 2x exactly -> Pearson r = 1, slope = 2, intercept = 0, Spearman
    rho = 1 (also perfectly monotonic). A perfect linear fit with n > 2
    always gives p = 0 (the t-statistic is infinite) — a fact independent
    of scipy's own implementation, not just "whatever scipy says"."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    result = S.correlation_pass_rate_vs_time(x, y)
    assert result is not None
    assert result.n == 5
    assert result.pearson_r == pytest.approx(1.0)
    assert result.pearson_p == pytest.approx(0.0, abs=1e-9)
    assert result.spearman_rho == pytest.approx(1.0)
    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(0.0, abs=1e-9)


def test_correlation_perfect_negative_linear_relationship():
    """y = -2x + 12 -> Pearson r = -1, slope = -2, intercept = 12."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([10.0, 8.0, 6.0, 4.0, 2.0])
    result = S.correlation_pass_rate_vs_time(x, y)
    assert result.pearson_r == pytest.approx(-1.0)
    assert result.spearman_rho == pytest.approx(-1.0)
    assert result.slope == pytest.approx(-2.0)
    assert result.intercept == pytest.approx(12.0)


def test_correlation_drops_nan_pairs_before_computing():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, None])
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 999.0])
    result = S.correlation_pass_rate_vs_time(x, y)
    assert result.n == 4  # the row with a missing x is dropped, not treated as 0
    assert result.pearson_r == pytest.approx(1.0)


def test_correlation_none_when_fewer_than_three_points():
    x = pd.Series([1.0, 2.0])
    y = pd.Series([2.0, 4.0])
    assert S.correlation_pass_rate_vs_time(x, y) is None


def test_correlation_none_when_x_is_constant():
    """Every point has the same planning time -> no meaningful fit/slope."""
    x = pd.Series([5.0, 5.0, 5.0])
    y = pd.Series([50.0, 60.0, 70.0])
    assert S.correlation_pass_rate_vs_time(x, y) is None


def test_correlation_empty_series():
    assert S.correlation_pass_rate_vs_time(pd.Series(dtype=float), pd.Series(dtype=float)) is None
