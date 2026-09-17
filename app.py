"""hypocxe-platform — navigation shell.

This file wires the pages together and does nothing else. All calculation
lives in engine/; see CLAUDE.md.
"""
from __future__ import annotations

import streamlit as st

from components.db import get_engine
from engine.config import DEFAULT_DB_URL
from engine.storage import ensure_daily_backup, recent_activity

st.set_page_config(
    page_title="HypoCxe Platform",
    page_icon="🧬",
    layout="wide",
)

PAGES = {
    "Intake": [
        st.Page("pages/0_new_case.py", title="New Case", icon=":material/note_add:"),
        st.Page("pages/1_patient_data.py", title="Patient Data", icon=":material/folder_shared:"),
    ],
    "Analysis": [
        st.Page("pages/2_pass_rate.py", title="Pass Rate", icon=":material/task_alt:"),
        st.Page("pages/3_time_efficiency.py", title="Time Efficiency", icon=":material/timer:"),
        st.Page("pages/4_plan_quality.py", title="Plan Quality", icon=":material/workspace_premium:"),
        st.Page("pages/5_dvh_comparison.py", title="DVH Comparison", icon=":material/show_chart:"),
    ],
    "Data": [
        st.Page("pages/8_data_review.py", title="Data Review", icon=":material/fact_check:"),
        st.Page("pages/9_registry_export.py", title="Registry Export", icon=":material/download:"),
    ],
}

# Streamlit reruns this script on every page visit, but ensure_daily_backup()
# is a cheap no-op once today's backup already exists, so calling it
# unconditionally here (rather than trying to run it "once") is fine.
ensure_daily_backup(DEFAULT_DB_URL)

engine = get_engine()
with st.sidebar:
    st.divider()
    st.caption("Recent activity")
    activity = recent_activity(engine, limit=8)
    if activity.empty:
        st.caption("Nothing logged yet.")
    else:
        for _, row in activity.iterrows():
            ts = row["timestamp"]
            ts_label = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
            st.caption(f"{ts_label} — **{row['action']}** — {row['detail']}")

navigation = st.navigation(PAGES)
navigation.run()
