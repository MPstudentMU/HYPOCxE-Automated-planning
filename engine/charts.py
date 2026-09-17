"""Plotly figure construction.

Pages call these and render whatever comes back; no figure is ever built
inline in a page (see CLAUDE.md rule 3). Module 1's registry charts
(cases_per_regimen_chart, cases_per_planner_chart) use one flat accent
color — #1F5FAE, already the app's primary/Auto-plan color — since each is
a single-series count-by-category bar with no adjacent-hue pair to keep
distinguishable. The pass-rate charts below use PLAN_COLORS instead: three
fixed, always-the-same-plan-type colors (docs/analysis_manual_th_v2.md's
own color spec for Manual/Auto/Auto+Manual), never cycled or reassigned.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from engine.analysis import PASS_RATE_TARGETS
from engine.schemas import PlanType

PRIMARY = "#1F5FAE"
GRID = "#DCE5F0"  # matches the app's existing hairline/border color

# The plan-type color coding used throughout the platform (docs/
# analysis_manual_th_v2.md, Module 1's own color spec).
PLAN_COLORS = {
    PlanType.MANUAL.value: "#8A94A6",
    PlanType.AUTO.value: "#1F5FAE",
    PlanType.AUTO_MANUAL.value: "#14A38B",
}
PLAN_ORDER = [p.value for p in PlanType]

_TARGET_LINE_STYLE = {
    PlanType.AUTO: dict(color=PLAN_COLORS[PlanType.AUTO.value],
                        label="Auto straight-pass target", position="top left"),
    PlanType.AUTO_MANUAL: dict(color=PLAN_COLORS[PlanType.AUTO_MANUAL.value],
                               label="Auto+Manual target", position="bottom left"),
}

_BASE_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(color="#0F2742"),
)


def cases_per_regimen_chart(registry_df: pd.DataFrame) -> go.Figure:
    """One bar per dose regimen: how many cases use it."""
    counts = registry_df["dose_regimen"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=PRIMARY,
        text=counts.values,
        textposition="outside",
        hovertemplate="%{x}: %{y} case(s)<extra></extra>",
    ))
    fig.update_layout(
        title="Cases per dose regimen",
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Cases", showgrid=True, gridcolor=GRID, rangemode="tozero"),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def cases_per_planner_chart(registry_df: pd.DataFrame) -> go.Figure:
    """One bar per MP1 (primary planner): how many cases they've planned,
    most cases first. Horizontal — planner names can run longer than a
    vertical axis comfortably fits without rotating labels."""
    counts = registry_df["mp1"].value_counts().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        marker_color=PRIMARY,
        text=counts.values,
        textposition="outside",
        hovertemplate="%{y}: %{x} case(s)<extra></extra>",
    ))
    fig.update_layout(
        title="Cases per planner (MP1)",
        xaxis=dict(title="Cases", showgrid=True, gridcolor=GRID, rangemode="tozero"),
        yaxis=dict(title=None, showgrid=False),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def _add_target_lines(fig: go.Figure) -> None:
    """Dashed reference lines at the Auto (50%) and Auto+Manual (70%) pass
    rate targets, colored to match each plan's own bar color."""
    for plan_type, style in _TARGET_LINE_STYLE.items():
        target = PASS_RATE_TARGETS[plan_type]
        fig.add_hline(
            y=target, line_dash="dash", line_color=style["color"], line_width=1.5,
            annotation_text=f"{style['label']} ({target:g}%)",
            annotation_position=style["position"], annotation_font_color=style["color"],
        )


def pass_rate_grouped_bar_chart(per_patient_df: pd.DataFrame) -> go.Figure:
    """One grouped bar per patient — Manual/Auto/Auto+Manual %Pass Rate —
    with the two target reference lines."""
    fig = go.Figure()
    patients = sorted(per_patient_df["pt_no"].unique())
    for plan_type in PLAN_ORDER:
        plan_df = per_patient_df[per_patient_df["plan_type"] == plan_type].set_index("pt_no")
        y = [plan_df["pass_rate"].get(pt) for pt in patients]
        fig.add_trace(go.Bar(
            name=plan_type, x=patients, y=y, marker_color=PLAN_COLORS[plan_type],
            hovertemplate=f"{plan_type}: " + "%{y:.1f}%<extra></extra>",
        ))
    _add_target_lines(fig)
    fig.update_layout(
        title="Pass rate per patient",
        barmode="group",
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="% Pass Rate", showgrid=True, gridcolor=GRID, range=[0, 108]),
        # Below the plot, not above: a legend row above the plot area sits
        # right next to (and can overlap) the chart's own title.
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=40, b=50),
        plot_bgcolor=_BASE_LAYOUT["plot_bgcolor"],
        paper_bgcolor=_BASE_LAYOUT["paper_bgcolor"],
        font=_BASE_LAYOUT["font"],
    )
    return fig


def pass_rate_mean_sd_chart(cohort_df: pd.DataFrame) -> go.Figure:
    """One bar per plan type: cohort mean %Pass Rate with an SD error bar,
    same target reference lines. A plan with only one patient has an
    undefined SD (not zero) — the cohort table is the place that shows
    that honestly; here the error bar just doesn't extend for that bar."""
    df = cohort_df.set_index("plan_type")
    x = [p for p in PLAN_ORDER if p in df.index]
    means = [df.loc[p, "mean"] for p in x]
    sds = [0.0 if pd.isna(df.loc[p, "sd"]) else df.loc[p, "sd"] for p in x]
    fig = go.Figure(go.Bar(
        x=x, y=means, marker_color=[PLAN_COLORS[p] for p in x],
        error_y=dict(type="data", array=sds, visible=True, color="#0F2742"),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    _add_target_lines(fig)
    fig.update_layout(
        title="Cohort mean ± SD pass rate",
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="% Pass Rate", showgrid=True, gridcolor=GRID, rangemode="tozero"),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig
