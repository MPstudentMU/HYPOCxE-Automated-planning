"""Pass Rate — Module 2 (docs/analysis_manual_th_v2.md's own numbering
calls this "Module 1: Clinical Acceptability Rate").

Display only: engine.analysis.compute_pass_rate does the calculation,
engine.stats.mcnemar_manual_vs_auto the significance test, engine.charts
the figures. This file loads the pipeline, renders the settings, and lays
out the tables/charts.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner
from engine import analysis as A
from engine import charts as CH
from engine import corrections as C
from engine import imputation as I
from engine import stats as S
from engine.schemas import PlanType
from engine.storage import load_analysis_frame

st.title("Pass Rate")
st.caption("Module 2 — criteria pass rate by plan type")

engine = get_engine()
render_pending_banner(engine)

raw = load_analysis_frame(engine)
if raw.empty:
    st.info("No cases yet — add one in New Case (Module 0).")
    st.stop()

# --------------------------------------------------------------------------- #
# Equation card
# --------------------------------------------------------------------------- #

st.latex(r"\%\ \text{Pass Rate} = \frac{n_{\text{passed}}}{n_{\text{total}}} \times 100")
st.caption(
    "% Pass Rate = (จำนวน Goal ที่ผ่านเกณฑ์ / จำนวน Goal ทั้งหมด) × 100 — the denominator is "
    "whatever goals remain after the filters below; Status comes from the TPS "
    "(recomputed for any corrected goal). / The denominator is the goal count remaining "
    "after the filters below; Status is the TPS-reported value, recomputed for corrected goals."
)

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

st.subheader("Settings")
s1, s2, s3 = st.columns(3)
exclude_non_evaluable = s1.checkbox("Exclude non-evaluable goals", value=True,
                                    help="A goal with no AchievedValue — a TPS computation "
                                        "limit, not a plan-quality signal.")
exclude_no_priority = s2.checkbox("Exclude goals without priority", value=True,
                                  help="RayStation's 'no priority set' sentinel.")
matched_goals_only = s3.checkbox("Matched goals only", value=False,
                                 help="Restrict to goal_keys present in every one of a "
                                     "patient's plans, for a like-for-like comparison. "
                                     "Decide before the primary analysis whether to use "
                                     "this — it can change results for individual patients.")

pipeline_df = I.apply_straight_pass(C.apply_corrections(raw, engine))
result = A.compute_pass_rate(
    pipeline_df,
    exclude_non_evaluable=exclude_non_evaluable,
    exclude_no_priority=exclude_no_priority,
    matched_goals_only=matched_goals_only,
)

straight_pass_patients = sorted(
    pipeline_df.loc[pipeline_df["imputed_from_auto"], "pt_no"].unique()
)
if straight_pass_patients:
    st.caption(f"🔁 Straight-pass (Auto+Manual imputed from Auto): {', '.join(straight_pass_patients)}")

# --------------------------------------------------------------------------- #
# Per-patient table
# --------------------------------------------------------------------------- #

st.subheader("Per patient")


def _wide(per_patient: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = per_patient.pivot(index="pt_no", columns="plan_type", values=value_col)
    return wide.reindex(columns=[p.value for p in PlanType])


rates = _wide(result.per_patient, "pass_rate")
targets_met = _wide(result.per_patient, "target_met")

display = rates.copy()
for plan in display.columns:
    display[plan] = [
        f"{r:.1f}%" + (" ✅" if tm is True else " ⚠️" if tm is False else "")
        if pd.notna(r) else "—"
        for r, tm in zip(rates[plan], targets_met[plan])
    ]
st.dataframe(display, width="stretch")

# --------------------------------------------------------------------------- #
# Cohort summary
# --------------------------------------------------------------------------- #

st.subheader("Cohort")


def _f1(v) -> str:
    return "—" if pd.isna(v) else f"{v:.1f}"


def _f0(v) -> str:
    return "—" if pd.isna(v) else f"{v:.0f}"


cohort = result.cohort
cohort_display = pd.DataFrame({
    "Plan": cohort["plan_type"],
    "n": cohort["n"],
    "Mean": cohort["mean"].map(_f1),
    "SD": cohort["sd"].map(_f1),
    "Median": cohort["median"].map(_f1),
    "Min": cohort["min"].map(_f1),
    "Max": cohort["max"].map(_f1),
    "CV (%)": cohort["cv"].map(_f1),
    "Target (%)": cohort["target"].map(_f0),
    "n (patients)": cohort["n_patients"],
    "Target met (n)": cohort["n_target_met"].map(_f0),
    "Target met (%)": cohort["target_met_pct"].map(_f0),
})
# Streamlit's st.dataframe doesn't reliably honor a pandas Styler's
# na_rep/number format (it renders through its own table widget, not
# HTML) — format to plain strings ourselves rather than relying on it,
# same as the per-patient table above.
st.dataframe(cohort_display, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

st.plotly_chart(CH.pass_rate_grouped_bar_chart(result.per_patient), width="stretch")
st.plotly_chart(CH.pass_rate_mean_sd_chart(result.cohort), width="stretch")

# --------------------------------------------------------------------------- #
# Priority-1 breakdown
# --------------------------------------------------------------------------- #

with st.expander("Priority-1 only breakdown"):
    p1_rates = _wide(result.priority1, "pass_rate")
    p1_totals = _wide(result.priority1, "n_total")
    p1_display = p1_rates.copy()
    for plan in p1_display.columns:
        p1_display[plan] = [
            f"{r:.1f}% ({int(n)} goal{'s' if n != 1 else ''})" if pd.notna(r) else "—"
            for r, n in zip(p1_rates[plan], p1_totals[plan])
        ]
    st.dataframe(p1_display, width="stretch")

# --------------------------------------------------------------------------- #
# McNemar's test (Manual vs Auto, per goal)
# --------------------------------------------------------------------------- #

st.subheader("Manual vs Auto agreement (McNemar's test)")
mc = S.mcnemar_manual_vs_auto(pipeline_df)
if mc is None:
    st.info("No goals are pairable between Manual and Auto yet (same patient, same goal_key).")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Paired goals", mc.n_pairs)
    m2.metric("Manual PASS, Auto FAIL", mc.b)
    m3.metric("Manual FAIL, Auto PASS", mc.c)
    m4.metric("p-value", f"{mc.p_value:.4f}" if mc.p_value is not None else "—")
    st.caption(f"Method: {mc.method}"
              + (f" (χ² = {mc.statistic:.3f})" if mc.statistic is not None else "")
              + ". Tests whether Manual and Auto disagree in one direction more than the "
                "other on goals they both have — concordant pairs (both PASS or both FAIL) "
                "don't affect the result.")
