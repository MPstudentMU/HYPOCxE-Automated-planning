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

from engine.analysis import EFFICIENCY_BAND_RANGE, PASS_RATE_TARGETS
from engine.schemas import PlanType

PRIMARY = "#1F5FAE"
GRID = "#DCE5F0"  # matches the app's existing hairline/border color
AMBER = "#F2A33A"  # target-band highlight color, used across Module 2 and 3

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

# docs/analysis_manual_th_v2.md §3.7's shared chart formatting rules, applied
# to "every chart" in this module: -45° x-axis labels, automargin on both
# axes, wide bottom margin for long rotated labels, no in-plot title (the
# card header carries it instead — callers set that, not this dict).
# pages/4_plan_quality.py already has its own identical local copy (`LAYOUT`)
# predating this file; not touched here, but this is the same four values,
# so anything built from each looks like one consistent module.
SHARED_CHART_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=50, b=150),
    xaxis=dict(tickangle=-45, automargin=True),
    yaxis=dict(automargin=True),
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


# --------------------------------------------------------------------------- #
# Module 3 — Time Efficiency
# --------------------------------------------------------------------------- #

_LEGEND_BELOW = dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
_MARGIN_WITH_BOTTOM_LEGEND = dict(l=10, r=10, t=40, b=50)


def planning_time_bar_chart(per_patient_df: pd.DataFrame) -> go.Figure:
    """One grouped bar per patient — Manual/Auto/Auto+Manual planning time
    in minutes. An Auto or Auto+Manual bar whose %efficiency lands in the
    40-75% Target band gets an amber outline and a "↓NN%" label — the
    manual's own example format."""
    fig = go.Figure()
    patients = per_patient_df["pt_no"].tolist()

    fig.add_trace(go.Bar(
        name=PlanType.MANUAL.value, x=patients, y=per_patient_df["time_manual"].tolist(),
        marker_color=PLAN_COLORS[PlanType.MANUAL.value],
        hovertemplate=f"{PlanType.MANUAL.value}: " + "%{y:.1f} min<extra></extra>",
    ))

    for plan_type, time_col, pct_col, band_col in [
        (PlanType.AUTO.value, "time_auto", "pct_eff_auto", "band_auto"),
        (PlanType.AUTO_MANUAL.value, "time_auto_manual", "pct_eff_auto_manual", "band_auto_manual"),
    ]:
        in_target = [b == "Target" for b in per_patient_df[band_col]]
        line_colors = [AMBER if t else "rgba(0,0,0,0)" for t in in_target]
        line_widths = [2.5 if t else 0 for t in in_target]
        labels = [f"↓{p:.0f}%" if t and pd.notna(p) else ""
                 for t, p in zip(in_target, per_patient_df[pct_col])]
        fig.add_trace(go.Bar(
            name=plan_type, x=patients, y=per_patient_df[time_col].tolist(),
            marker=dict(color=PLAN_COLORS[plan_type], line=dict(color=line_colors, width=line_widths)),
            text=labels, textposition="outside",
            hovertemplate=f"{plan_type}: " + "%{y:.1f} min<extra></extra>",
        ))

    fig.update_layout(
        title="Planning time per patient",
        barmode="group",
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Minutes", showgrid=True, gridcolor=GRID, rangemode="tozero"),
        legend=_LEGEND_BELOW,
        margin=_MARGIN_WITH_BOTTOM_LEGEND,
        plot_bgcolor=_BASE_LAYOUT["plot_bgcolor"], paper_bgcolor=_BASE_LAYOUT["paper_bgcolor"],
        font=_BASE_LAYOUT["font"],
    )
    return fig


def time_efficiency_bar_chart(per_patient_df: pd.DataFrame) -> go.Figure:
    """%Time Efficiency per patient for Auto and Auto+Manual (Manual has
    none — it's the baseline the other two are measured against), with
    the 40-75% Target band shaded as a background rectangle."""
    fig = go.Figure()
    patients = per_patient_df["pt_no"].tolist()
    for plan_type, pct_col in [
        (PlanType.AUTO.value, "pct_eff_auto"),
        (PlanType.AUTO_MANUAL.value, "pct_eff_auto_manual"),
    ]:
        fig.add_trace(go.Bar(
            name=plan_type, x=patients, y=per_patient_df[pct_col].tolist(),
            marker_color=PLAN_COLORS[plan_type],
            hovertemplate=f"{plan_type}: " + "%{y:.1f}%<extra></extra>",
        ))

    low, high = EFFICIENCY_BAND_RANGE
    fig.add_hrect(y0=low, y1=high, fillcolor=AMBER, opacity=0.12, layer="below", line_width=0,
                 annotation_text=f"Target ({low:g}-{high:g}%)", annotation_position="top left",
                 annotation_font_color=AMBER)

    fig.update_layout(
        title="Time efficiency per patient",
        barmode="group",
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="% Time Efficiency", showgrid=True, gridcolor=GRID),
        legend=_LEGEND_BELOW,
        margin=_MARGIN_WITH_BOTTOM_LEGEND,
        plot_bgcolor=_BASE_LAYOUT["plot_bgcolor"], paper_bgcolor=_BASE_LAYOUT["paper_bgcolor"],
        font=_BASE_LAYOUT["font"],
    )
    return fig


_PLAN_MARKER_SYMBOLS = {
    PlanType.MANUAL.value: "circle",
    PlanType.AUTO.value: "square",
    PlanType.AUTO_MANUAL.value: "diamond",
}


def pass_rate_vs_time_scatter(scatter_df: pd.DataFrame, correlation=None) -> go.Figure:
    """%Pass Rate (y) vs planning time (x), colored and shaped by plan
    type, with a linear fit line when `correlation` (an
    engine.stats.CorrelationResult) is given. Per the manual's own
    caption, this is for spotting a trend, not for inference — points
    from the same patient (several plan types) aren't independent; show
    that caveat next to this chart, not just the r/ρ numbers."""
    fig = go.Figure()
    for plan_type in PLAN_ORDER:
        plan_df = scatter_df[scatter_df["plan_type"] == plan_type]
        if plan_df.empty:
            continue
        fig.add_trace(go.Scatter(
            name=plan_type, x=plan_df["planning_time_min"], y=plan_df["pass_rate"],
            mode="markers",
            marker=dict(color=PLAN_COLORS[plan_type], symbol=_PLAN_MARKER_SYMBOLS[plan_type],
                       size=10, line=dict(color="white", width=1)),
            text=plan_df["pt_no"],
            hovertemplate=f"{plan_type} — " + "%{text}<br>%{x:.1f} min, %{y:.1f}%<extra></extra>",
        ))

    if correlation is not None and not scatter_df.empty:
        x_range = [scatter_df["planning_time_min"].min(), scatter_df["planning_time_min"].max()]
        y_fit = [correlation.slope * x + correlation.intercept for x in x_range]
        fig.add_trace(go.Scatter(
            x=x_range, y=y_fit, mode="lines", line=dict(color="#0F2742", dash="dash", width=1.5),
            hoverinfo="skip", showlegend=False,
        ))
        annotation_text = "<br>".join([
            f"Pearson r = {correlation.pearson_r:.2f} (p = {correlation.pearson_p:.3f})",
            f"Spearman ρ = {correlation.spearman_rho:.2f} (p = {correlation.spearman_p:.3f})",
            f"n = {correlation.n}",
        ])
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left", yanchor="top",
            text=annotation_text, showarrow=False, align="left",
            font=dict(size=12, color="#0F2742"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor=GRID, borderwidth=1,
        )

    fig.update_layout(
        title="Pass rate vs planning time",
        xaxis=dict(title="Planning time (min)", showgrid=True, gridcolor=GRID),
        yaxis=dict(title="% Pass Rate", showgrid=True, gridcolor=GRID),
        legend=_LEGEND_BELOW,
        margin=_MARGIN_WITH_BOTTOM_LEGEND,
        plot_bgcolor=_BASE_LAYOUT["plot_bgcolor"], paper_bgcolor=_BASE_LAYOUT["paper_bgcolor"],
        font=_BASE_LAYOUT["font"],
    )
    return fig


# --------------------------------------------------------------------------- #
# Module 3 — DVH Comparison (§3.7)
# --------------------------------------------------------------------------- #


def _wrap_axis_label(roi: str, goal_text: str, max_len: int = 16) -> str:
    """§3.7's shared rule: a long label is line-broken between the
    structure name and the metric, each side truncated at 16 characters."""
    roi_part = roi if len(roi) <= max_len else roi[:max_len]
    goal_part = goal_text if len(goal_text) <= max_len else goal_text[:max_len]
    return f"{roi_part}<br>{goal_part}"


def dvh_boxplot(dvh_frame: pd.DataFrame, *, value_col: str = "value", unit_label: str = "") -> go.Figure:
    """One box per (goal, plan) — grouped by goal on the x-axis, colored
    by plan, boxmean='sd', every point shown and jittered. Call this with
    rows from a single unit_category only (or all-Normalised rows, which
    share a comparable scale by construction) — engine.analysis
    .compute_dvh_frame keeps unit_category distinct precisely so nothing
    here has to guess whether mixing is safe.
    """
    if dvh_frame.empty:
        return go.Figure()

    labels = {
        goal_key: _wrap_axis_label(g["roi"].iloc[0], g["goal_text"].iloc[0])
        for goal_key, g in dvh_frame.groupby("goal_key")
    }

    fig = go.Figure()
    for plan_type in PLAN_ORDER:
        plan_df = dvh_frame[dvh_frame["plan"] == plan_type]
        if plan_df.empty:
            continue
        fig.add_trace(go.Box(
            x=plan_df["goal_key"].map(labels), y=plan_df[value_col], name=plan_type,
            marker_color=PLAN_COLORS[plan_type], boxmean="sd", boxpoints="all",
            jitter=0.4, pointpos=0, text=plan_df["patient"],
            hovertemplate="%{text}: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(boxmode="group", yaxis_title=unit_label, **SHARED_CHART_LAYOUT)
    return fig


def quality_index_spread_chart(qi_table: pd.DataFrame, cohort_table: pd.DataFrame) -> go.Figure:
    """One Quality Index boxplot per plan type, each labeled with its own
    SD on the x-axis — §3.7's "Quality-index Spread Boxplot". qi_table /
    cohort_table are engine.priority_filter.quality_index_table() /
    cohort_summary()'s own output; this only draws what they computed.
    """
    fig = go.Figure()
    for _, row in cohort_table.iterrows():
        plan = row["plan"]
        vals = qi_table.loc[qi_table["plan"] == plan, "quality_index"].dropna()
        sd_label = "—" if pd.isna(row["sd"]) else f"{row['sd']:.1f}"
        fig.add_trace(go.Box(
            y=vals, name=f"{plan}<br>SD = {sd_label}", boxpoints="all",
            jitter=0.3, pointpos=0, marker_color=PLAN_COLORS.get(plan, PRIMARY), boxmean="sd",
        ))
    fig.update_layout(showlegend=False, yaxis_title="Quality Index (%)", **SHARED_CHART_LAYOUT)
    return fig


def structure_group_radar_chart(radar_table_df: pd.DataFrame) -> go.Figure:
    """Mean base score (-2..+3) by structure group, one overlay per plan
    — §3.7's Radar Chart. radar_table_df is
    engine.priority_filter.radar_table()'s own output.
    """
    fig = go.Figure()
    for plan_type in PLAN_ORDER:
        sub = radar_table_df[radar_table_df["plan"] == plan_type]
        if sub.empty:
            continue
        theta = sub["structure_group"].tolist()
        r = sub["mean_base"].tolist()
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=theta + theta[:1], name=plan_type,
            line_color=PLAN_COLORS[plan_type], fill="toself", opacity=0.45,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[-2, 3])), margin=dict(l=40, r=40, t=50, b=40))
    return fig
