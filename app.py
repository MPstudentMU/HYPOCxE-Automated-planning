"""hypocxe-platform — navigation shell.

This file wires the pages together, gates the app behind a shared
passphrase, and does nothing else. All calculation lives in engine/; see
CLAUDE.md.

Access control: one shared passphrase (engine.auth), not per-user accounts
— see the README's "Access control" section for what this is (and isn't)
meant to protect against. Both gates below (login, then the one-time
"Entered by" prompt) sit at the very top of this file, before PAGES/
st.navigation are even built, and each calls st.stop() itself — so no page
content can render without authenticated=True, checked once here rather
than per-page.
"""
from __future__ import annotations

import streamlit as st

from components.db import get_engine
from engine.auth import verify_passphrase
from engine.config import DEFAULT_DB_URL
from engine.storage import ensure_daily_backup, recent_activity

st.set_page_config(
    page_title="HypoCxe Platform",
    page_icon="🧬",
    layout="wide",
)


def _expected_passphrase_hash() -> str | None:
    try:
        return st.secrets.get("app_passphrase_hash")
    except Exception:
        # No .streamlit/secrets.toml at all — st.secrets raises rather
        # than behaving like an empty mapping in that case.
        return None


def _render_login() -> None:
    st.title("HypoCxe Platform")
    st.caption("Enter the shared passphrase to continue.")

    expected_hash = _expected_passphrase_hash()
    if not expected_hash:
        st.error(
            "No passphrase is configured. Copy `.streamlit/secrets.toml.example` to "
            "`.streamlit/secrets.toml` and set `app_passphrase_hash` — see the README's "
            "Access control section."
        )
        st.stop()

    _, center, _ = st.columns([1, 1, 1])
    with center:
        with st.form("login_form"):
            passphrase = st.text_input("Passphrase", type="password")
            submitted = st.form_submit_button("Enter", type="primary", width="stretch")
        if submitted:
            if verify_passphrase(passphrase, expected_hash):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect passphrase.")
    st.stop()


def _render_entered_by_gate() -> None:
    st.title("HypoCxe Platform")
    st.caption(
        "One-time for this session: who's using the app? Shown on saves, corrections and "
        "exports as \"Entered by\" — not a user account, and not remembered after this "
        "session ends."
    )
    _, center, _ = st.columns([1, 1, 1])
    with center:
        with st.form("entered_by_form"):
            name = st.text_input("Entered by")
            submitted = st.form_submit_button("Continue", type="primary", width="stretch")
        if submitted:
            if not name.strip():
                st.error("Enter a name to continue.")
            else:
                st.session_state["entered_by"] = name.strip()
                st.rerun()
    st.stop()


if not st.session_state.get("authenticated", False):
    _render_login()

if "entered_by" not in st.session_state:
    _render_entered_by_gate()

PAGES = {
    "Intake": [
        st.Page("pages/0_new_case.py", title="New Case", icon=":material/note_add:"),
        st.Page("pages/0b_batch_upload.py", title="Batch Upload", icon=":material/upload_file:"),
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
    if st.button("Log out", icon=":material/logout:", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.caption(f"Entered by: **{st.session_state['entered_by']}**")
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
