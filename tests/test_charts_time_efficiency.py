"""Tests for the Module 3 (Time Efficiency) charts in engine/charts.py."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from engine import charts as CH
from engine.analysis import EFFICIENCY_BAND_RANGE
from engine.stats import CorrelationResult


@pytest.fixture
def per_patient_df():
    return pd.DataFrame([
        dict(pt_no="Pt1", time_manual=100.0, time_auto=25.0, time_auto_manual=25.0,
            pct_eff_auto=75.0, pct_eff_auto_manual=75.0, band_auto="Target", band_auto_manual="Target"),
        dict(pt_no="Pt2", time_manual=100.0, time_auto=70.0, time_auto_manual=50.0,
            pct_eff_auto=30.0, pct_eff_auto_manual=50.0, band_auto="Below", band_auto_manual="Target"),
        dict(pt_no="Pt5", time_manual=float("nan"), time_auto=30.0, time_auto_manual=30.0,
            pct_eff_auto=float("nan"), pct_eff_auto_manual=float("nan"), band_auto=None, band_auto_manual=None),
    ])


# --------------------------------------------------------------------------- #
# planning_time_bar_chart
# --------------------------------------------------------------------------- #


def test_planning_time_bar_chart_has_three_traces(per_patient_df):
    fig = CH.planning_time_bar_chart(per_patient_df)
    assert isinstance(fig, go.Figure)
    assert [t.name for t in fig.data] == ["Manual", "Auto", "Auto+Manual"]


def test_planning_time_bar_chart_values(per_patient_df):
    fig = CH.planning_time_bar_chart(per_patient_df)
    by_name = {t.name: list(t.y) for t in fig.data}
    assert by_name["Manual"][0] == 100.0
    assert by_name["Auto"] == [25.0, 70.0, 30.0]
    assert by_name["Auto+Manual"] == [25.0, 50.0, 30.0]


def test_planning_time_bar_chart_amber_outline_only_on_target_band(per_patient_df):
    fig = CH.planning_time_bar_chart(per_patient_df)
    auto_trace = next(t for t in fig.data if t.name == "Auto")
    # Pt1 (Target) amber, Pt2 (Below) not, Pt5 (None/NaN) not
    assert auto_trace.marker.line.color[0] == CH.AMBER
    assert auto_trace.marker.line.color[1] != CH.AMBER
    assert auto_trace.marker.line.color[2] != CH.AMBER
    assert auto_trace.marker.line.width[0] > 0
    assert auto_trace.marker.line.width[1] == 0


def test_planning_time_bar_chart_target_band_label_format(per_patient_df):
    fig = CH.planning_time_bar_chart(per_patient_df)
    am_trace = next(t for t in fig.data if t.name == "Auto+Manual")
    # Pt1: 75% (Target) -> labeled; Pt2: 50% (also Target for Auto+Manual) -> labeled
    assert list(am_trace.text) == ["↓75%", "↓50%", ""]


def test_planning_time_bar_chart_manual_never_gets_a_target_label(per_patient_df):
    fig = CH.planning_time_bar_chart(per_patient_df)
    manual_trace = next(t for t in fig.data if t.name == "Manual")
    assert manual_trace.text is None


# --------------------------------------------------------------------------- #
# time_efficiency_bar_chart
# --------------------------------------------------------------------------- #


def test_time_efficiency_bar_chart_two_traces_auto_and_auto_manual(per_patient_df):
    fig = CH.time_efficiency_bar_chart(per_patient_df)
    assert [t.name for t in fig.data] == ["Auto", "Auto+Manual"]


def test_time_efficiency_bar_chart_values(per_patient_df):
    fig = CH.time_efficiency_bar_chart(per_patient_df)
    by_name = {t.name: list(t.y) for t in fig.data}
    assert by_name["Auto"][:2] == pytest.approx([75.0, 30.0])
    assert by_name["Auto+Manual"][:2] == pytest.approx([75.0, 50.0])


def test_time_efficiency_bar_chart_has_target_band_shape(per_patient_df):
    fig = CH.time_efficiency_bar_chart(per_patient_df)
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 1
    low, high = EFFICIENCY_BAND_RANGE
    assert rects[0].y0 == low
    assert rects[0].y1 == high


# --------------------------------------------------------------------------- #
# pass_rate_vs_time_scatter
# --------------------------------------------------------------------------- #


@pytest.fixture
def scatter_df():
    return pd.DataFrame([
        dict(pt_no="Pt1", plan_type="Manual", pass_rate=100.0, planning_time_min=100.0),
        dict(pt_no="Pt1", plan_type="Auto", pass_rate=100.0, planning_time_min=25.0),
        dict(pt_no="Pt1", plan_type="Auto+Manual", pass_rate=100.0, planning_time_min=25.0),
        dict(pt_no="Pt5", plan_type="Auto", pass_rate=75.0, planning_time_min=30.0),
        dict(pt_no="Pt5", plan_type="Auto+Manual", pass_rate=75.0, planning_time_min=30.0),
    ])


def test_scatter_has_one_trace_per_plan_type_present(scatter_df):
    fig = CH.pass_rate_vs_time_scatter(scatter_df, correlation=None)
    assert {t.name for t in fig.data} == {"Manual", "Auto", "Auto+Manual"}


def test_scatter_uses_distinct_marker_symbol_per_plan_type(scatter_df):
    fig = CH.pass_rate_vs_time_scatter(scatter_df, correlation=None)
    symbols = {t.name: t.marker.symbol for t in fig.data}
    assert len(set(symbols.values())) == 3  # all different


def test_scatter_point_values(scatter_df):
    fig = CH.pass_rate_vs_time_scatter(scatter_df, correlation=None)
    auto_trace = next(t for t in fig.data if t.name == "Auto")
    assert list(auto_trace.x) == [25.0, 30.0]
    assert list(auto_trace.y) == [100.0, 75.0]


def test_scatter_no_fit_line_or_annotation_without_correlation(scatter_df):
    fig = CH.pass_rate_vs_time_scatter(scatter_df, correlation=None)
    assert len(fig.data) == 3  # one per plan type, no separate fit-line trace
    assert len(fig.layout.annotations) == 0


def test_scatter_adds_fit_line_and_annotation_with_correlation(scatter_df):
    correlation = CorrelationResult(n=5, pearson_r=0.5, pearson_p=0.02,
                                    spearman_rho=0.4, spearman_p=0.03, slope=1.5, intercept=10.0)
    fig = CH.pass_rate_vs_time_scatter(scatter_df, correlation=correlation)
    assert len(fig.data) == 4  # 3 plan-type traces + 1 fit-line trace
    fit_trace = fig.data[-1]
    assert fit_trace.mode == "lines"
    assert len(fig.layout.annotations) == 1
    text = fig.layout.annotations[0].text
    assert "Pearson r = 0.50" in text
    assert "Spearman" in text
    assert "n = 5" in text


def test_scatter_fit_line_endpoints_match_slope_and_intercept(scatter_df):
    correlation = CorrelationResult(n=5, pearson_r=1.0, pearson_p=0.0,
                                    spearman_rho=1.0, spearman_p=0.0, slope=2.0, intercept=1.0)
    fig = CH.pass_rate_vs_time_scatter(scatter_df, correlation=correlation)
    fit_trace = fig.data[-1]
    x0, x1 = fit_trace.x
    y0, y1 = fit_trace.y
    assert y0 == pytest.approx(2.0 * x0 + 1.0)
    assert y1 == pytest.approx(2.0 * x1 + 1.0)


def test_scatter_empty_df_does_not_crash():
    fig = CH.pass_rate_vs_time_scatter(pd.DataFrame(columns=["pt_no", "plan_type", "pass_rate",
                                                              "planning_time_min"]), correlation=None)
    assert isinstance(fig, go.Figure)
