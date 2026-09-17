"""Pre-save pending-goals preview — the New Case / Batch Upload flows'
shared editor.

Different from components/correction_editor.py: that one writes directly to
engine.storage.GoalCorrection, which needs a real goal_id (a saved row).
Here nothing is saved yet — parsing just produced a list of PlanFrames in
memory — so render_pending_preview() collects what the user enters and
hands it back as a DataFrame keyed by (plan_type, goal_key), and
apply_pending_corrections() turns those rows into real
engine.corrections.record_correction() calls once save_case()/replace_plan()
has given every goal a real goal_id. Rendering only, plus that one
straightforward application step both pages/0_new_case.py and
pages/0b_batch_upload.py need identically (originally written inline in
pages/0_new_case.py; factored out here once a second caller needed it).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy.engine import Engine

from engine import corrections as C
from engine.schemas import CorrectionField, CorrectionSource, CorrectionStatus, PlanFrame
from engine.storage import load_analysis_frame


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


def apply_pending_corrections(engine: Engine, pt_no: str, edited_pending: pd.DataFrame, *,
                              corrected_by: str) -> tuple[int, list[str], list[str]]:
    """Turn render_pending_preview()'s edited rows into real
    engine.corrections.record_correction() calls, once save_case() (or
    replace_plan()) has given this patient's goals real goal_ids. Returns
    (saved_count, warnings, errors) for the caller to display.

    Looks goal_ids up via load_analysis_frame() filtered to `pt_no`, keyed
    by (plan_type, goal_key) — the same identity render_pending_preview's
    rows already carry from the in-memory PlanFrames they were built from.
    A row whose goal_id can't be found (shouldn't happen if save happened
    first) is silently skipped rather than raising — there's nothing
    sensible to correct if the goal itself never made it into the database.
    """
    saved, errors, warns = 0, [], []
    if edited_pending.empty:
        return saved, warns, errors

    df = load_analysis_frame(engine)
    df = df[df["pt_no"] == pt_no]
    goal_id_by_key = {(r.plan_type, r.goal_key): r.goal_id for r in df.itertuples()}

    for _, row in edited_pending.iterrows():
        goal_id = goal_id_by_key.get((row["plan_type"], row["goal_key"]))
        if goal_id is None:
            continue
        label = f"{row['plan_type']} / {row['roi']}"
        source = CorrectionSource(row["source"]) if row["source"] else None

        if row["_missing_value"] and (row["confirmed_not_evaluable"] or pd.notna(row["value"])):
            try:
                status = (CorrectionStatus.CONFIRMED_NOT_EVALUABLE
                         if row["confirmed_not_evaluable"] else CorrectionStatus.CORRECTED)
                _, warn = C.record_correction(
                    engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                    status=status, source=source, reason=row["reason"],
                    corrected_by=corrected_by,
                    corrected_value_display=(
                        None if row["confirmed_not_evaluable"] else float(row["value"])
                    ),
                )
                saved += 1
                if warn:
                    warns.append(f"{label}: {warn}")
            except C.CorrectionValidationError as exc:
                errors.append(f"{label}: {exc}")

        if row["_missing_priority"] and pd.notna(row["priority"]):
            try:
                C.record_correction(
                    engine, goal_id=goal_id, field=CorrectionField.PRIORITY,
                    status=CorrectionStatus.CORRECTED, source=source,
                    reason=row["reason"], corrected_by=corrected_by,
                    corrected_value_display=row["priority"],
                )
                saved += 1
            except C.CorrectionValidationError as exc:
                errors.append(f"{label}: {exc}")

    return saved, warns, errors
