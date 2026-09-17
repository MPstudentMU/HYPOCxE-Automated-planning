"""
Module 4 – Plan Quality Score (with Priority Filtering)

Integration points (adapt the two functions in the ADAPTER section to your
v2 engine names; everything below them is ready to use):
  * load_goal_scores()   -> goal-level scoring DataFrame (see
                            engine/priority_filter.py for required columns)
  * load_critical_alerts() -> DataFrame of PTV coverage alerts (unfiltered)
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner
from engine.priority_filter import (
    PLAN_ORDER,
    PrioritySelection,
    category_breakdown,
    cohort_summary,
    paired_consistency,
    quality_index_table,
    radar_table,
)

PLAN_COLORS = {"Manual": "#8A94A6", "Auto": "#1F5FAE", "Auto+Manual": "#14A38B"}
CATEGORY_COLORS = {"OAR Sparing": "#1F5FAE", "Target Coverage": "#14A38B",
                   "Hot-spot Limit": "#E0A526"}
LAYOUT = dict(margin=dict(l=20, r=20, t=50, b=150),
              xaxis=dict(tickangle=-45, automargin=True),
              yaxis=dict(automargin=True))

# --------------------------------------------------------------------------- #
# ADAPTER – connect to your existing engine
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Scoring plans with Criteria V0…")
def load_goal_scores(settings_key: str) -> pd.DataFrame:
    # Full, unfiltered scoring is cached; the priority filter is applied
    # afterwards, so changing the filter never re-scores anything.
    from engine import analysis
    return analysis.compute_goal_scores()          # <- adapt name


@st.cache_data
def load_critical_alerts(settings_key: str) -> pd.DataFrame:
    from engine import analysis
    return analysis.compute_critical_alerts()      # <- adapt name


# --------------------------------------------------------------------------- #
# Priority filter widget
# --------------------------------------------------------------------------- #
ALL = "All Priorities"
LEVEL_OPTIONS = {"Priority 1": 1, "Priority 2": 2, "Priority 3": 3}
KEY = "m4_priority_filter"
PREV_KEY = f"{KEY}_prev"


def _normalise_filter() -> None:
    """Keep 'All Priorities' and specific levels mutually exclusive."""
    cur = list(st.session_state.get(KEY) or [])
    prev = st.session_state.get(PREV_KEY, [ALL])
    if not cur:                                   # nothing selected -> All
        new = [ALL]
    elif ALL in cur and ALL not in prev:          # user just clicked All
        new = [ALL]
    elif ALL in cur:                              # user added a level
        new = [c for c in cur if c != ALL]
    elif set(cur) == set(LEVEL_OPTIONS):          # 1+2+3 == All
        new = [ALL]
    else:
        new = cur
    st.session_state[KEY] = new
    st.session_state[PREV_KEY] = new


def render_priority_filter() -> PrioritySelection:
    st.session_state.setdefault(KEY, [ALL])
    st.session_state.setdefault(PREV_KEY, [ALL])
    st.segmented_control(
        "Priority filter",
        options=[ALL, *LEVEL_OPTIONS],
        selection_mode="multi",
        key=KEY,
        on_change=_normalise_filter,
        help="Scores and Quality Index are recalculated using only the "
             "clinical goals of the selected priority level(s).",
    )
    chosen = st.session_state[KEY]
    if ALL in chosen:
        return PrioritySelection.all()
    return PrioritySelection(tuple(LEVEL_OPTIONS[c] for c in chosen))


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_qi_per_patient(qi: pd.DataFrame, label: str) -> go.Figure:
    data = qi.dropna(subset=["quality_index"])
    show_text = data["patient"].nunique() <= 8
    fig = px.bar(
        data, x="patient", y="quality_index", color="plan", barmode="group",
        category_orders={"plan": PLAN_ORDER}, color_discrete_map=PLAN_COLORS,
        text=data["quality_index"].round(1) if show_text else None,
        hover_data={"score": ":.1f", "max_score": ":.1f", "n_goals": True},
        labels={"quality_index": f"Quality Index (%) – {label}",
                "patient": "Patient", "plan": "Plan"},
    )
    fig.update_layout(**LAYOUT)
    fig.add_hline(y=0, line_color="#999", line_width=1)
    return fig


def chart_qi_spread(qi: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for _, row in summary.iterrows():
        plan = row["plan"]
        vals = qi.loc[qi["plan"] == plan, "quality_index"].dropna()
        sd = "—" if pd.isna(row["sd"]) else f"{row['sd']:.1f}"
        fig.add_trace(go.Box(
            y=vals, name=f"{plan}<br>SD = {sd}", boxpoints="all",
            jitter=0.3, pointpos=0, marker_color=PLAN_COLORS[plan],
            boxmean="sd"))
    fig.update_layout(**LAYOUT, showlegend=False,
                      yaxis_title="Quality Index (%)")
    return fig


def chart_breakdown(bd: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for cat, color in CATEGORY_COLORS.items():
        sub = bd[bd["category"] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=[sub["patient"], sub["plan"].astype(str)],
            y=sub["weighted_score"], name=cat, marker_color=color))
    fig.update_layout(**LAYOUT, barmode="relative",
                      yaxis_title="Weighted score")
    return fig


def chart_radar(rd: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for plan in PLAN_ORDER:
        sub = rd[rd["plan"] == plan]
        if sub.empty:
            continue
        theta = sub["structure_group"].tolist()
        r = sub["mean_base"].tolist()
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=theta + theta[:1], name=plan,
            line_color=PLAN_COLORS[plan], fill="toself", opacity=0.45))
    fig.update_layout(polar=dict(radialaxis=dict(range=[-2, 3])),
                      margin=dict(l=40, r=40, t=50, b=40))
    return fig


def _fmt(v, nd=1):
    return "—" if pd.isna(v) else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("Module 4 · Plan Quality Score")

render_pending_banner(get_engine())

settings_key = str(st.session_state.get("analysis_settings", "default"))
goal_scores = load_goal_scores(settings_key)
alerts = load_critical_alerts(settings_key)

# Critical Alert is a safety check on PTV coverage -> never priority-filtered
if alerts is not None and not alerts.empty:
    failed = alerts[alerts["status"] == "FAIL (Unacceptable)"]
    if not failed.empty:
        st.error(
            f"⛔ Critical Alert (independent of priority filter): "
            f"{len(failed)} plan(s) with PTV coverage < 95% – "
            + ", ".join(f"{r.patient} {r.plan}" for r in failed.itertuples()))

selection = render_priority_filter()

# Equation card follows the filter
st.latex(
    r"\text{Quality Index}_{\,%s}\;(\%%)=\frac{\sum_{%s} \text{Base}_i"
    r"\times \text{Multiplier}_i}{\sum_{%s} \text{MaxBase}_i \times"
    r" \text{Multiplier}_i}\times 100"
    % (r"\text{" + selection.label + "}", selection.latex_set,
       selection.latex_set))
st.caption(
    f"Showing **{selection.label}**. Criteria V0 scoring is unchanged; only "
    "the set of goals being summed changes. Plans with no scorable goals in "
    "the selected priority are shown as “—”.")

qi = quality_index_table(goal_scores, selection)
summary = cohort_summary(qi)

if qi["quality_index"].notna().sum() == 0:
    st.info(f"No scorable goals found for {selection.label}.")
    st.stop()

# ---- Table --------------------------------------------------------------- #
st.subheader(f"Quality Index per patient – {selection.label}")
wide = (qi.assign(plan=qi["plan"].astype(str))
          .pivot_table(index="patient", columns="plan",
                       values="quality_index", aggfunc="first", dropna=False)
          .reindex(columns=PLAN_ORDER))
wide.columns = list(wide.columns)
st.dataframe(wide.style.format(_fmt), width="stretch")

with st.expander("Score / Max score / number of goals"):
    detail = qi[["patient", "plan", "n_goals", "score", "max_score",
                 "quality_index"]].assign(plan=lambda d: d["plan"].astype(str))
    st.dataframe(detail.style.format(
        {"score": _fmt, "max_score": _fmt, "quality_index": _fmt}),
        width="stretch", hide_index=True)

st.dataframe(summary.style.format(
    {c: _fmt for c in ["mean", "sd", "median", "min", "max", "cv_pct"]}),
    width="stretch", hide_index=True)

# ---- Charts -------------------------------------------------------------- #
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(f"**Quality Index per patient – {selection.label}**")
    st.plotly_chart(chart_qi_per_patient(qi, selection.label),
                    width="stretch")
with c2:
    st.markdown(f"**Quality Index spread – {selection.label}**")
    st.plotly_chart(chart_qi_spread(qi, summary), width="stretch")

c3, c4 = st.columns(2)
with c3:
    st.markdown(f"**Score breakdown by category – {selection.label}**")
    st.plotly_chart(chart_breakdown(category_breakdown(goal_scores, selection)),
                    width="stretch")
with c4:
    st.markdown(f"**Mean base score by structure – {selection.label}**")
    rd = radar_table(goal_scores, selection)
    if rd.empty:
        st.caption("No structure groups in this selection.")
    else:
        st.plotly_chart(chart_radar(rd), width="stretch")

# ---- Consistency tests --------------------------------------------------- #
st.markdown(f"**Paired consistency tests – {selection.label}**")
tests = paired_consistency(qi)
st.dataframe(tests.style.format(
    {"var_ratio": lambda v: _fmt(v, 2), "pm_t": lambda v: _fmt(v, 2),
     "pm_p": lambda v: _fmt(v, 3), "wilcoxon_p": lambda v: _fmt(v, 3)}),
    width="stretch", hide_index=True)
st.caption("Filtering reduces the number of goals per plan; with small n, "
           "report these tests descriptively.")

# ---- Goal-level detail --------------------------------------------------- #
with st.expander(f"Goal-level scoring detail – {selection.label}"):
    from engine.priority_filter import filter_goal_scores
    st.dataframe(filter_goal_scores(goal_scores, selection),
                 width="stretch", hide_index=True)

# Make the active filter available to the export page (RunInfo sheet)
st.session_state["m4_priority_filter_label"] = selection.label
