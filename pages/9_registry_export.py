"""Registry Export — Module 9.

Builds and downloads one multi-sheet Excel workbook covering every module
(RunInfo, M1_Patients, M2_PassRate, M2_Detail, M3_Time, M4_QualitySummary,
M4_GoalScores, M4_Consistency, M5_DVHConsistency, Alerts, DataCorrections).
Display only: engine.export.build_analysis_workbook() does all the work,
including logging the run to analysis_runs — this page just wires a button
to it. HN is always masked here; there is no unmask option on an export
(CLAUDE.md rule 4 — Module 1's on-screen checkbox is the only exception).

The active priority filter is picked up from whatever Module 4 (Plan
Quality) last set in st.session_state, so the export matches what the user
was actually looking at; it defaults to "All Priorities" if Module 4 was
never visited this session.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner
from engine.export import build_analysis_workbook
from engine.priority_filter import PrioritySelection
from engine.storage import load_analysis_frame

st.title("Registry Export")
st.caption("Module 9 — masked registry export")

engine = get_engine()
render_pending_banner(engine)

if load_analysis_frame(engine).empty:
    st.info("No cases yet — add one in New Case (Module 0).")
    st.stop()

levels = st.session_state.get("m4_priority_filter_levels")
selection = PrioritySelection(levels) if levels else PrioritySelection.all()
filter_label = st.session_state.get("m4_priority_filter_label", selection.label)

st.markdown(
    "Produces one `.xlsx` workbook with a sheet per module — patient registry "
    "(HN masked), pass rate, time efficiency, plan quality/consistency, DVH "
    "consistency, critical alerts, and the full correction audit trail — plus "
    "a `RunInfo` sheet recording the engine/criteria version, this run's "
    "settings, and the export timestamp. Every export is logged."
)
st.caption(f"Plan quality sheets (M4_QualitySummary, M4_Consistency) will use the "
          f"priority filter last set on Plan Quality: **{filter_label}**.")

if st.button("Prepare export", icon=":material/table_view:"):
    with st.spinner("Building workbook…"):
        st.session_state["export_workbook"] = build_analysis_workbook(
            engine, priority_selection=selection,
        )
        st.session_state["export_workbook_built_at"] = datetime.now()

if "export_workbook" in st.session_state:
    built_at = st.session_state["export_workbook_built_at"]
    st.success(f"Workbook ready — built {built_at:%Y-%m-%d %H:%M:%S}.")
    st.download_button(
        "Download workbook",
        data=st.session_state["export_workbook"],
        file_name=f"hypocxe_export_{built_at:%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
    )
    st.caption("HN is masked in every sheet — this file is safe to move off the "
              "local network. See the README's PDPA section.")
