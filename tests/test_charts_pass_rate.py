"""Tests for the pass-rate charts in engine/charts.py."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from engine import charts as CH
from engine.analysis import PASS_RATE_TARGETS
from engine.schemas import PlanType


@pytest.fixture
def per_patient_df():
    return pd.DataFrame([
        dict(pt_no="Pt1", plan_type="Manual", pass_rate=100.0),
        dict(pt_no="Pt1", plan_type="Auto", pass_rate=80.0),
        dict(pt_no="Pt1", plan_type="Auto+Manual", pass_rate=90.0),
        dict(pt_no="Pt2", plan_type="Manual", pass_rate=90.0),
        dict(pt_no="Pt2", plan_type="Auto", pass_rate=40.0),
        dict(pt_no="Pt2", plan_type="Auto+Manual", pass_rate=75.0),
    ])


@pytest.fixture
def cohort_df():
    return pd.DataFrame([
        dict(plan_type="Manual", mean=95.0, sd=7.07),
        dict(plan_type="Auto", mean=60.0, sd=28.28),
        dict(plan_type="Auto+Manual", mean=82.5, sd=10.6),
    ])


def test_grouped_bar_chart_has_one_trace_per_plan_type(per_patient_df):
    fig = CH.pass_rate_grouped_bar_chart(per_patient_df)
    assert isinstance(fig, go.Figure)
    assert [t.name for t in fig.data] == ["Manual", "Auto", "Auto+Manual"]


def test_grouped_bar_chart_values_correct(per_patient_df):
    fig = CH.pass_rate_grouped_bar_chart(per_patient_df)
    by_name = {t.name: dict(zip(t.x, t.y)) for t in fig.data}
    assert by_name["Manual"] == {"Pt1": 100.0, "Pt2": 90.0}
    assert by_name["Auto"] == {"Pt1": 80.0, "Pt2": 40.0}
    assert by_name["Auto+Manual"] == {"Pt1": 90.0, "Pt2": 75.0}


def test_grouped_bar_chart_uses_plan_colors(per_patient_df):
    fig = CH.pass_rate_grouped_bar_chart(per_patient_df)
    colors = {t.name: t.marker.color for t in fig.data}
    assert colors["Manual"] == "#8A94A6"
    assert colors["Auto"] == "#1F5FAE"
    assert colors["Auto+Manual"] == "#14A38B"


def test_grouped_bar_chart_has_two_target_lines(per_patient_df):
    fig = CH.pass_rate_grouped_bar_chart(per_patient_df)
    hlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(hlines) == 2
    y_values = sorted(s.y0 for s in hlines)
    assert y_values == [PASS_RATE_TARGETS[PlanType.AUTO], PASS_RATE_TARGETS[PlanType.AUTO_MANUAL]]


def test_grouped_bar_chart_single_axis(per_patient_df):
    assert "yaxis2" not in CH.pass_rate_grouped_bar_chart(per_patient_df).to_dict()["layout"]


def test_mean_sd_chart_values_and_error_bars(cohort_df):
    fig = CH.pass_rate_mean_sd_chart(cohort_df)
    bar = fig.data[0]
    assert list(bar.x) == ["Manual", "Auto", "Auto+Manual"]
    assert list(bar.y) == pytest.approx([95.0, 60.0, 82.5])
    assert list(bar.error_y.array) == pytest.approx([7.07, 28.28, 10.6])


def test_mean_sd_chart_handles_undefined_sd_gracefully():
    """A single-patient plan's NaN SD must not crash the chart — it just
    renders with a zero-length error bar (the cohort table is where the
    undefined-SD distinction is shown honestly)."""
    df = pd.DataFrame([dict(plan_type="Manual", mean=100.0, sd=float("nan"))])
    fig = CH.pass_rate_mean_sd_chart(df)
    assert fig.data[0].error_y.array[0] == 0.0


def test_mean_sd_chart_has_two_target_lines(cohort_df):
    fig = CH.pass_rate_mean_sd_chart(cohort_df)
    hlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(hlines) == 2
