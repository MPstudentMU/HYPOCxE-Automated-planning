"""Pre-save pending-goals preview — the New Case upload flow's editor.

Different from components/correction_editor.py: that one writes directly to
engine.storage.GoalCorrection, which needs a real goal_id (a saved row).
Here nothing is saved yet — parsing just produced a list of PlanFrames in
memory — so this collects what the user enters and hands it back as a
DataFrame keyed by (plan_type, goal_key); pages/0_new_case.py applies it via
engine.corrections.record_correction() right after save_case() gives it real
goal_ids. Rendering only.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import corrections as C
from engine.schemas import CorrectionSource, PlanFrame


def _flagged_rows(plan_frames: list[PlanFrame]) -> pd.DataFrame:
    rows = []
    for frame in plan_frames:
        for goal in frame.goals:
            missing_value = goal.achieved_value is None
            missing_priority = goal.priority is None
            if not (missing_value or missing_priority):
                continue
            unit = C.unit_for_goal_type(goal.goal_type)
            rows.append(dict(
                plan_type=frame.plan_type.value,
                goal_key=goal.goal_key,
                roi=goal.roi,
                goal_text=goal.goal_text,
                goal_type=goal.goal_type,
                issue=" & ".join(filter(None, [
                    "missing value" if missing_value else None,
                    "missing priority" if missing_priority else None,
                ])),
                unit=unit,
                _missing_value=missing_value,
                _missing_priority=missing_priority,
                value=None,
                priority=None,
                confirmed_not_evaluable=False,
                source=None,
                reason="",
            ))
    return pd.DataFrame(rows)


def render_pending_preview(plan_frames: list[PlanFrame], *, key: str) -> pd.DataFrame:
    """Shows goal counts per plan, and — if anything is flagged — an
    editable table for it. Returns the edited rows (empty DataFrame if
    nothing was flagged) for the caller to turn into corrections after
    save_case().
    """
    counts = pd.Series({frame.plan_type.value: len(frame.goals) for frame in plan_frames})
    st.subheader("Preview")
    cols = st.columns(len(counts) or 1)
    for col, (plan_type, n) in zip(cols, counts.items()):
        col.metric(plan_type, n)

    flagged = _flagged_rows(plan_frames)
    if flagged.empty:
        st.success("Every goal has an achieved value and a priority — nothing to review.")
        return flagged

    n = len(flagged)
    st.warning(f"⚠️ {n} goal{'s' if n != 1 else ''} need review. Fill in what you can below "
              "before confirming — anything left blank can still be resolved later in "
              "Data Review (Module 8).", icon="⚠️")

    display_cols = ["plan_type", "roi", "goal_text", "issue", "unit", "value",
                    "priority", "confirmed_not_evaluable", "source", "reason"]
    edited = st.data_editor(
        flagged[display_cols],
        key=f"{key}_preview_editor",
        hide_index=True,
        disabled=["plan_type", "roi", "goal_text", "issue", "unit"],
        column_config={
            "plan_type": "Plan",
            "roi": "ROI",
            "goal_text": "Goal",
            "issue": "Issue",
            "unit": "Unit",
            "value": st.column_config.NumberColumn("Value", help="In the unit shown"),
            "priority": st.column_config.SelectboxColumn("Priority", options=[1, 2, 3]),
            "confirmed_not_evaluable": st.column_config.CheckboxColumn("Confirmed not evaluable"),
            "source": st.column_config.SelectboxColumn(
                "Source", options=[s.value for s in CorrectionSource]
            ),
            "reason": st.column_config.TextColumn("Reason"),
        },
    )
    # carry the hidden matching/flag columns through for the caller
    return edited.assign(
        goal_key=flagged["goal_key"].values,
        _missing_value=flagged["_missing_value"].values,
        _missing_priority=flagged["_missing_priority"].values,
    )
