"""DVH Comparison — placeholder.

Module 5. Not yet implemented. Display only: all calculation belongs in
engine/ (see CLAUDE.md).
"""
from __future__ import annotations

import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner

st.title("DVH Comparison")
st.caption("Module 5 — dose–volume histogram comparison")

render_pending_banner(get_engine())

st.info("Not yet implemented.", icon=":material/construction:")
