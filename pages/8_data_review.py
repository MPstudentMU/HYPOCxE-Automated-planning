"""Data Review — Module 8.

Every goal, across every patient, still needing a manual correction —
filterable by patient and plan. Same editor as the New Case upload preview
(components/correction_editor.py); this page just doesn't scope it to one
just-uploaded case. Display only: all validation/persistence is
engine/corrections.py.
"""
from __future__ import annotations

import streamlit as st

from components.correction_editor import render_correction_editor
from components.db import get_engine
from components.pending_banner import render_pending_banner
from engine import corrections as C
from engine.schemas import PlanType
from engine.storage import load_analysis_frame

st.title("Data Review")
st.caption("Module 8 — completeness, corrections and data errors")

engine = get_engine()
render_pending_banner(engine)

df = load_analysis_frame(engine)
if df.empty:
    st.info("No cases yet — add one in New Case (Module 0).")
    st.stop()

corrected_df = C.apply_corrections(df, engine)

st.subheader("Filter")
f1, f2 = st.columns(2)
patients = f1.multiselect("Patient", sorted(corrected_df["pt_no"].unique()))
plans = f2.multiselect("Plan", [p.value for p in PlanType])

filtered = corrected_df
if patients:
    filtered = filtered[filtered["pt_no"].isin(patients)]
if plans:
    filtered = filtered[filtered["plan_type"].isin(plans)]

st.divider()
render_correction_editor(engine, filtered, key="data_review")
