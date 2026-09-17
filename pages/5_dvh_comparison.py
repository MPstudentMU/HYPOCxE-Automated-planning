"""DVH Comparison — Module 5 (part of the manual's own Module 3, §3.7 —
"DVH Consistency" alongside pages/4_plan_quality.py's Quality Index).

Display only: engine.analysis.compute_dvh_frame/compute_dvh_consistency do
the calculation, engine.stats.pitman_morgan_test the paired-variance test,
engine.charts the figures. The Quality-index spread boxplot and structure-
group radar chart reuse engine.priority_filter's own
quality_index_table/cohort_summary/radar_table (and
engine.analysis.compute_goal_scores) — not recomputed here.
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
from engine.priority_filter import PrioritySelection, cohort_summary, quality_index_table, radar_table
from engine.schemas import StructureClass
from engine.storage import load_analysis_frame

st.title("DVH Comparison")
st.caption("Module 5 — dose–volume histogram comparison")

engine = get_engine()
render_pending_banner(engine)

raw = load_analysis_frame(engine)
if raw.empty:
    st.info("No cases yet — add one in New Case (Module 0).")
    st.stop()

pipeline_df = I.apply_straight_pass(C.apply_corrections(raw, engine))
dvh_all = A.compute_dvh_frame(pipeline_df)

if dvh_all.empty:
    st.info("No goals with an achieved value yet — nothing to plot.")
    st.stop()

n_straight_pass = int(pipeline_df.loc[pipeline_df["imputed_from_auto"], "pt_no"].nunique())
if n_straight_pass:
    st.caption(f"🔁 {n_straight_pass} patient(s) in view have Auto+Manual imputed from Auto "
              "(straight-pass — no manual touch-up was needed).")

# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #

UNIT_OPTIONS = ["Dose (Gy)", "%Volume", "Volume (cc)", "Normalised (% of Goal)"]

s1, s2, s3 = st.columns(3)
structure_class = s1.radio("Structure class", [StructureClass.TARGET.value, StructureClass.OAR.value])

roi_options = sorted(dvh_all.loc[dvh_all["structure_class"] == structure_class, "roi"].unique())
if not roi_options:
    st.info(f"No {structure_class.lower()} goals with an achieved value in this cohort.")
    st.stop()
roi = s2.selectbox("ROI", roi_options)

unit = s3.selectbox("Metric unit", UNIT_OPTIONS)

by_structure_roi = dvh_all[(dvh_all["structure_class"] == structure_class) & (dvh_all["roi"] == roi)]

if unit == "Normalised (% of Goal)":
    filtered = by_structure_roi.dropna(subset=["normalised_pct"])
    value_col, unit_label = "normalised_pct", "% of Goal"
else:
    filtered = by_structure_roi[by_structure_roi["unit_category"] == unit]
    value_col, unit_label = "value", unit

if filtered.empty:
    st.info(f"No '{unit}' goals for {roi} in this cohort.")
    st.stop()

# --------------------------------------------------------------------------- #
# Boxplot
# --------------------------------------------------------------------------- #

st.subheader(f"{roi} — {unit}")
st.plotly_chart(CH.dvh_boxplot(filtered, value_col=value_col, unit_label=unit_label), width="stretch")

# --------------------------------------------------------------------------- #
# Consistency table
# --------------------------------------------------------------------------- #

st.subheader("Consistency")

consistency = A.compute_dvh_consistency(filtered, value_col=value_col)


def _f1(v) -> str:
    return "—" if pd.isna(v) else f"{v:.1f}"


def _f3(v) -> str:
    return "—" if pd.isna(v) else f"{v:.3f}"


def _n_label(n, insufficient) -> str:
    return f"{int(n)} ⚠️ insufficient n" if insufficient else str(int(n))


def _bool_label(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return "✅ Yes" if v else "No"


rows = []
for _, r in consistency.iterrows():
    rows.append({
        "Goal": f"{r['roi']} — {r['goal_text']}",
        "n (Manual)": _n_label(r["n_manual"], r["insufficient_n_manual"]),
        "Mean (Manual)": _f1(r["mean_manual"]), "SD (Manual)": _f1(r["sd_manual"]),
        "CV% (Manual)": _f1(r["cv_manual"]),
        "n (Auto)": _n_label(r["n_auto"], r["insufficient_n_auto"]),
        "Mean (Auto)": _f1(r["mean_auto"]), "SD (Auto)": _f1(r["sd_auto"]),
        "CV% (Auto)": _f1(r["cv_auto"]),
        "n (Auto+Manual)": _n_label(r["n_auto_manual"], r["insufficient_n_auto_manual"]),
        "Mean (Auto+Manual)": _f1(r["mean_auto_manual"]), "SD (Auto+Manual)": _f1(r["sd_auto_manual"]),
        "CV% (Auto+Manual)": _f1(r["cv_auto_manual"]),
        "SD(Auto) < SD(Manual)": _bool_label(r["sd_lower_than_manual_auto"]),
        "Pitman-Morgan p (Auto)": _f3(r["pm_p_auto"]),
        "Wilcoxon p (Auto)": _f3(r["wilcoxon_p_auto"]),
        "SD(Auto+Manual) < SD(Manual)": _bool_label(r["sd_lower_than_manual_auto_manual"]),
        "Pitman-Morgan p (Auto+Manual)": _f3(r["pm_p_auto_manual"]),
        "Wilcoxon p (Auto+Manual)": _f3(r["wilcoxon_p_auto_manual"]),
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

m1, m2 = st.columns(2)
n_lower_auto = int((consistency["sd_lower_than_manual_auto"] == True).sum())  # noqa: E712
n_comparable_auto = int(consistency["sd_lower_than_manual_auto"].notna().sum())
m1.metric("Metrics with SD(Auto) < SD(Manual)",
         f"{n_lower_auto} / {n_comparable_auto}" if n_comparable_auto else "—")
n_lower_am = int((consistency["sd_lower_than_manual_auto_manual"] == True).sum())  # noqa: E712
n_comparable_am = int(consistency["sd_lower_than_manual_auto_manual"].notna().sum())
m2.metric("Metrics with SD(Auto+Manual) < SD(Manual)",
         f"{n_lower_am} / {n_comparable_am}" if n_comparable_am else "—")

if filtered["patient"].nunique() < 10:
    st.caption("⚠️ Low power: with this few patients, the variance tests above have limited "
              "power to detect a real difference. Report SD/CV/the count above descriptively "
              "until the full sample size is reached.")

# --------------------------------------------------------------------------- #
# Quality-index spread boxplot + structure-group radar chart
# (reused from engine.priority_filter / engine.analysis — not recomputed)
# --------------------------------------------------------------------------- #

st.subheader("Plan quality — cohort spread")
goal_scores = A.compute_goal_scores(pipeline_df)
selection = PrioritySelection.all()
qi = quality_index_table(goal_scores, selection)
qi_summary = cohort_summary(qi)
rd = radar_table(goal_scores, selection)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Quality Index spread — All Priorities**")
    st.plotly_chart(CH.quality_index_spread_chart(qi, qi_summary), width="stretch")
with c2:
    st.markdown("**Mean base score by structure — All Priorities**")
    if rd.empty:
        st.caption("No structure groups scored yet.")
    else:
        st.plotly_chart(CH.structure_group_radar_chart(rd), width="stretch")
st.caption("Same Quality Index/radar calculation as Module 4 (Plan Quality) — see that page to "
          "filter by priority level.")
