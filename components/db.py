"""Shared, cached database engine for every page.

Streamlit-specific glue only — engine/storage.py itself has no Streamlit
dependency (that's why it's fully testable under plain pytest); this is the
one place a Streamlit cache decorator wraps it for the app.
"""
from __future__ import annotations

import streamlit as st

from engine.config import DEFAULT_DB_URL
from engine.storage import init_db


@st.cache_resource
def get_engine():
    """The app's one SQLAlchemy engine, created once per server process."""
    return init_db(DEFAULT_DB_URL)
