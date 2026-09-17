"""Patient Data — Module 1.

The on-screen registry: one row per patient, HN masked by default (unmask
behind an explicit checkbox — never a default, never carried into a chart,
export, or any Module 2-5 analysis table; see CLAUDE.md rule 4). Filters,
cohort counts, and two small bar charts. Display only: engine.export builds
the table and engine.charts builds the figures.
"""
from __future__ import annotations

import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner
from engine.charts import cases_per_planner_chart, cases_per_regimen_chart
from engine.export import load_registry_frame, mask_hn

st.title("Patient Data")
st.caption("Module 1 — patient and plan records")

engine = get_engine()
render_pending_banner(engine)

df = load_registry_frame(engine)
if df.empty:
    st.info("No cases yet — add one in New Case (Module 0).")
    st.stop()

st.subheader("Filter")
f1, f2, f3, f4, f5 = st.columns(5)
regimens = f1.multiselect("Dose regimen", sorted(df["dose_regimen"].unique()))
sib_choice = f2.selectbox("SIB boost", ["All", "Yes", "No"])
rooms = f3.multiselect("Tx room", sorted(df["tx_room"].unique()))
ros = f4.multiselect("RO", sorted(df["ro"].unique()))
mp1s = f5.multiselect("MP1", sorted(df["mp1"].unique()))

filtered = df
if regimens:
    filtered = filtered[filtered["dose_regimen"].isin(regimens)]
if sib_choice != "All":
    filtered = filtered[filtered["sib_boost"] == (sib_choice == "Yes")]
if rooms:
    filtered = filtered[filtered["tx_room"].isin(rooms)]
if ros:
    filtered = filtered[filtered["ro"].isin(ros)]
if mp1s:
    filtered = filtered[filtered["mp1"].isin(mp1s)]

st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Cases", len(filtered))
c2.metric("With SIB boost", int(filtered["sib_boost"].sum()))
c3.metric("Goals pending review", int(filtered["pending_count"].sum()))

st.divider()

unmask = st.checkbox(
    "Unmask HN",
    help="Shows the full HN on screen for this view only — never saved, exported, "
        "charted, or carried into any analysis table.",
)
display = filtered.copy()
display["HN"] = display["hn"] if unmask else display["hn"].map(mask_hn)
display["Straight-pass"] = display["straight_pass"].map(
    lambda v: "🔁 Yes" if v is True else ("No" if v is False else "—")
)
display["Alert"] = display["pending_count"].map(
    lambda n: "✅ OK" if n == 0 else f"⚠️ {n} pending"
)
display = display.rename(columns={
    "pt_no": "Patient", "dose_regimen": "Regimen", "sib_boost": "SIB",
    "tx_room": "Tx room", "mp1": "MP1", "mp2": "MP2", "ro": "RO",
    "plans_uploaded": "Plans uploaded", "created_at": "Upload date",
})

st.dataframe(
    display[["Patient", "HN", "Regimen", "SIB", "Tx room", "MP1", "MP2", "RO",
             "Plans uploaded", "Straight-pass", "Alert", "Upload date"]],
    hide_index=True,
    width="stretch",
)

st.divider()

st.subheader("Cohort")
chart_col1, chart_col2 = st.columns(2)
chart_col1.plotly_chart(cases_per_regimen_chart(filtered), width="stretch")
chart_col2.plotly_chart(cases_per_planner_chart(filtered), width="stretch")
