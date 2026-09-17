"""Plotly figure construction.

Pages call these and render whatever comes back; no figure is ever built
inline in a page (see CLAUDE.md rule 3). One flat accent color per chart
here — #1F5FAE ("medical blue"), already the app's primary/Auto-plan color
in pages/4_plan_quality.py — rather than a new categorical palette: these
are single-series count-by-category bars, so there's no adjacent-hue pair
to keep distinguishable, just one consistent brand color across the app.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PRIMARY = "#1F5FAE"
GRID = "#DCE5F0"  # matches the app's existing hairline/border color

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
