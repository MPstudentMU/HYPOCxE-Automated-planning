"""Tests for the Module 3/DVH-Comparison charts in engine/charts.py."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from engine import charts as CH


# --------------------------------------------------------------------------- #
# _wrap_axis_label — the 16-char truncation + line-break rule
# --------------------------------------------------------------------------- #


def test_wrap_axis_label_short_strings_unchanged():
    assert CH._wrap_axis_label("PTV", "V95 >= 95%") == "PTV<br>V95 >= 95%"


def test_wrap_axis_label_truncates_each_side_at_16():
    roi = "A" * 20
    goal = "B" * 20
    label = CH._wrap_axis_label(roi, goal)
    roi_part, goal_part = label.split("<br>")
    assert len(roi_part) == 16
    assert len(goal_part) == 16
    assert roi_part == "A" * 16
    assert goal_part == "B" * 16


def test_wrap_axis_label_exactly_16_not_truncated_further():
    roi = "C" * 16
    label = CH._wrap_axis_label(roi, "x")
    assert label.split("<br>")[0] == roi


# --------------------------------------------------------------------------- #
# dvh_boxplot
# --------------------------------------------------------------------------- #


@pytest.fixture
def dvh_frame():
    return pd.DataFrame([
        dict(patient="Pt1", plan="Manual", goal_key="g1", roi="Bladder",
            goal_text="D0.03cc <= 4725 cGy", value=47.15),
        dict(patient="Pt2", plan="Manual", goal_key="g1", roi="Bladder",
            goal_text="D0.03cc <= 4725 cGy", value=46.90),
        dict(patient="Pt1", plan="Auto", goal_key="g1", roi="Bladder",
            goal_text="D0.03cc <= 4725 cGy", value=47.10),
        dict(patient="Pt2", plan="Auto", goal_key="g1", roi="Bladder",
            goal_text="D0.03cc <= 4725 cGy", value=46.50),
    ])


def test_dvh_boxplot_one_trace_per_plan_present(dvh_frame):
    fig = CH.dvh_boxplot(dvh_frame, unit_label="Dose (Gy)")
    assert isinstance(fig, go.Figure)
    assert {t.name for t in fig.data} == {"Manual", "Auto"}


def test_dvh_boxplot_uses_plan_colors(dvh_frame):
    fig = CH.dvh_boxplot(dvh_frame)
    colors = {t.name: t.marker.color for t in fig.data}
    assert colors["Manual"] == CH.PLAN_COLORS["Manual"]
    assert colors["Auto"] == CH.PLAN_COLORS["Auto"]


def test_dvh_boxplot_shows_boxmean_sd_and_all_points(dvh_frame):
    fig = CH.dvh_boxplot(dvh_frame)
    for trace in fig.data:
        assert trace.boxmean == "sd"
        assert trace.boxpoints == "all"


def test_dvh_boxplot_values_correct(dvh_frame):
    fig = CH.dvh_boxplot(dvh_frame)
    manual_trace = next(t for t in fig.data if t.name == "Manual")
    assert sorted(manual_trace.y) == pytest.approx([46.90, 47.15])


def test_dvh_boxplot_x_axis_uses_wrapped_labels(dvh_frame):
    fig = CH.dvh_boxplot(dvh_frame)
    x_values = set(fig.data[0].x)
    assert x_values == {"Bladder<br>D0.03cc <= 4725 "}


def test_dvh_boxplot_empty_frame_returns_empty_figure():
    fig = CH.dvh_boxplot(pd.DataFrame(columns=["patient", "plan", "goal_key", "roi", "goal_text", "value"]))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_dvh_boxplot_uses_shared_layout_margins(dvh_frame):
    fig = CH.dvh_boxplot(dvh_frame)
    layout = fig.to_dict()["layout"]
    assert layout["margin"] == dict(l=20, r=20, t=50, b=150)
    assert layout["xaxis"]["tickangle"] == -45
    assert layout["xaxis"]["automargin"] is True
    assert layout["yaxis"]["automargin"] is True


# --------------------------------------------------------------------------- #
# quality_index_spread_chart
# --------------------------------------------------------------------------- #


@pytest.fixture
def qi_and_cohort():
    qi = pd.DataFrame([
        dict(patient="Pt1", plan="Manual", quality_index=80.0),
        dict(patient="Pt2", plan="Manual", quality_index=90.0),
        dict(patient="Pt1", plan="Auto", quality_index=60.0),
        dict(patient="Pt2", plan="Auto", quality_index=65.0),
    ])
    cohort = pd.DataFrame([
        dict(plan="Manual", sd=7.071068),
        dict(plan="Auto", sd=3.535534),
        dict(plan="Auto+Manual", sd=float("nan")),
    ])
    return qi, cohort


def test_quality_index_spread_chart_one_box_per_plan(qi_and_cohort):
    qi, cohort = qi_and_cohort
    fig = CH.quality_index_spread_chart(qi, cohort)
    assert len(fig.data) == 3


def test_quality_index_spread_chart_labels_include_sd(qi_and_cohort):
    qi, cohort = qi_and_cohort
    fig = CH.quality_index_spread_chart(qi, cohort)
    names = [t.name for t in fig.data]
    assert "Manual<br>SD = 7.1" in names
    assert "Auto<br>SD = 3.5" in names
    assert "Auto+Manual<br>SD = —" in names  # NaN SD shown honestly, not fabricated


def test_quality_index_spread_chart_values(qi_and_cohort):
    qi, cohort = qi_and_cohort
    fig = CH.quality_index_spread_chart(qi, cohort)
    manual_trace = next(t for t in fig.data if t.name.startswith("Manual"))
    assert sorted(manual_trace.y) == pytest.approx([80.0, 90.0])


# --------------------------------------------------------------------------- #
# structure_group_radar_chart
# --------------------------------------------------------------------------- #


@pytest.fixture
def radar_df():
    return pd.DataFrame([
        dict(structure_group="Target", plan="Manual", mean_base=2.0),
        dict(structure_group="Bladder", plan="Manual", mean_base=1.0),
        dict(structure_group="Target", plan="Auto", mean_base=1.5),
        dict(structure_group="Bladder", plan="Auto", mean_base=0.5),
    ])


def test_structure_group_radar_chart_one_trace_per_plan_present(radar_df):
    fig = CH.structure_group_radar_chart(radar_df)
    assert {t.name for t in fig.data} == {"Manual", "Auto"}


def test_structure_group_radar_chart_closes_the_loop(radar_df):
    """The first theta/r point is repeated at the end to close the polygon."""
    fig = CH.structure_group_radar_chart(radar_df)
    manual_trace = next(t for t in fig.data if t.name == "Manual")
    assert manual_trace.theta[0] == manual_trace.theta[-1]
    assert manual_trace.r[0] == manual_trace.r[-1]


def test_structure_group_radar_chart_skips_plans_with_no_data():
    df = pd.DataFrame([dict(structure_group="Target", plan="Manual", mean_base=2.0)])
    fig = CH.structure_group_radar_chart(df)
    assert len(fig.data) == 1


def test_structure_group_radar_chart_radial_range_matches_base_score_scale():
    """Base scores run -2..+3 (Criteria V0) — the radial axis must show
    that whole range, not auto-scale to just the plotted values."""
    df = pd.DataFrame([dict(structure_group="Target", plan="Manual", mean_base=1.0)])
    fig = CH.structure_group_radar_chart(df)
    assert fig.layout.polar.radialaxis.range == (-2, 3)
