"""Pending-review banner, shown at the top of every module page.

Rendering only — the count itself is engine.corrections.pending_count.
"""
from __future__ import annotations

import streamlit as st

from engine import corrections as C


def render_pending_banner(engine) -> None:
    """Warn if any goal, for any patient, is still missing an achieved
    value or a priority. Silent when there's nothing pending."""
    n = C.pending_count(engine)
    if n:
        goal_word = "goal" if n == 1 else "goals"
        st.warning(
            f"⚠️ {n} {goal_word} across the cohort still need review — "
            "see **Data Review** (Module 8).",
            icon="⚠️",
        )
