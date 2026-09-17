"""Time Efficiency — Module 3 (docs/analysis_manual_th_v2.md's own
numbering calls this "Module 2: Time Efficiency").

Display only: engine.analysis.compute_time_efficiency does the
calculation, engine.stats.correlation_pass_rate_vs_time the correlation,
engine.charts the figures. This file loads the pipeline, renders the
equation card, tables and charts.
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
from engine.storage import load_analysis_frame

st.title("Time Efficiency")
st.caption("Module 3 — planning time by plan type")

engine = get_engine()
render_pending_banner(engine)

raw = load_analysis_frame(engine)
if raw.empty:
    st.info("No cases yet — add one in New Case (Module 0).")
    st.stop()

pipeline_df = I.apply_straight_pass(C.apply_corrections(raw, engine))

# --------------------------------------------------------------------------- #
# Equation card
# --------------------------------------------------------------------------- #

e1, e2 = st.columns(2)
with e1:
    st.latex(r"\%\ \text{Efficiency}_{\text{Auto}} = "
            r"\frac{T_{\text{Manual}} - T_{\text{Auto}}}{T_{\text{Manual}}} \times 100")
with e2:
    st.latex(r"\%\ \text{Efficiency}_{\text{Auto+Manual}} = "
            r"\frac{T_{\text{Manual}} - T_{\text{Auto+Manual}}}{T_{\text{Manual}}} \times 100")
st.caption(
    "% Time Efficiency = [(เวลา Manual − เวลา แผน) / เวลา Manual] × 100 — Auto+Manual ใช้เวลาสะสม "
    "(สร้างแผนอัตโนมัติ + ปรับแก้); ผู้ป่วย Straight-pass ค่าทั้งสองจะเท่ากัน. / Auto+Manual uses the "
    "cumulative time (automated planning + manual touch-up); a straight-pass patient's Auto and "
    "Auto+Manual efficiency are identical, since no touch-up time was added."
)

# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #

result = A.compute_time_efficiency(pipeline_df)


def _f1(v) -> str:
    return "—" if pd.isna(v) else f"{v:.1f}"


def _fpct(v) -> str:
    return "—" if pd.isna(v) else f"{v:.1f}%"


def _fband(v) -> str:
    if pd.isna(v):
        return "—"
    icon = {"Below": "⬇️", "Target": "✅", "Exceeds": "🚀"}.get(v, "")
    return f"{icon} {v}".strip()


st.subheader("Per patient")
per_patient_display = pd.DataFrame({
    "Patient": result.per_patient["pt_no"],
    "Manual (min)": result.per_patient["time_manual"].map(_f1),
    "Auto (min)": result.per_patient["time_auto"].map(_f1),
    "Auto+Manual (min)": result.per_patient["time_auto_manual"].map(_f1),
    "Auto % Eff": result.per_patient["pct_eff_auto"].map(_fpct),
    "Auto Band": result.per_patient["band_auto"].map(_fband),
    "Auto+Manual % Eff": result.per_patient["pct_eff_auto_manual"].map(_fpct),
    "Auto+Manual Band": result.per_patient["band_auto_manual"].map(_fband),
})
# Plain strings, not a pandas Styler: st.dataframe doesn't reliably honor a
# Styler's na_rep/number format (it renders through its own table widget,
# not Styler's HTML) — same pattern as pages/2_pass_rate.py.
st.dataframe(per_patient_display, hide_index=True, width="stretch")

st.subheader("Cohort")
cohort_display = pd.DataFrame({
    "Plan": result.cohort["plan_type"],
    "n": result.cohort["n"],
    "Mean % Eff": result.cohort["mean"].map(_f1),
    "SD": result.cohort["sd"].map(_f1),
    "Below (n)": result.cohort["n_below"],
    "Target (n)": result.cohort["n_target"],
    "Exceeds (n)": result.cohort["n_exceeds"],
})
st.dataframe(cohort_display, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

st.plotly_chart(CH.planning_time_bar_chart(result.per_patient), width="stretch")
st.plotly_chart(CH.time_efficiency_bar_chart(result.per_patient), width="stretch")

# --------------------------------------------------------------------------- #
# Pass rate vs planning time
# --------------------------------------------------------------------------- #

st.subheader("Pass rate vs planning time")

pass_rate_result = A.compute_pass_rate(pipeline_df)  # this page's own defaults, independent
                                                      # of whatever toggles Module 2 has set
scatter_df = A.pass_rate_vs_time_frame(pass_rate_result, pipeline_df)
correlation = S.correlation_pass_rate_vs_time(scatter_df["planning_time_min"], scatter_df["pass_rate"])

st.plotly_chart(CH.pass_rate_vs_time_scatter(scatter_df, correlation), width="stretch")

if correlation is None:
    st.info("Not enough paired (planning time, pass rate) points yet for a correlation.")

st.caption(
    "จุดที่อยู่บริเวณมุมซ้ายบน (เวลาน้อย ผ่านเกณฑ์สูง) แสดงถึงความคุ้มค่าสูงสุดของกระบวนการอัตโนมัติ "
    "ค่าสหสัมพันธ์คำนวณจากทุกจุดของทุกแผนรวมกัน ซึ่งไม่เป็นอิสระต่อกันอย่างสมบูรณ์ (ผู้ป่วยรายเดียวกันให้ข้อมูลหลายจุด) "
    "จึงควรใช้เพื่อการสำรวจแนวโน้มมากกว่าการอนุมานเชิงสถิติ / A point toward the upper-left (low time, "
    "high pass rate) represents the best value from automation. Correlation is computed across every "
    "point from every plan type pooled together — these are not fully independent (the same patient "
    "contributes several points) — so use this to explore a trend, not for statistical inference."
)
