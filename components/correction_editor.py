"""Reusable pending-goals correction editor.

Used by pages/0_new_case.py (one patient, right after upload) and
pages/8_data_review.py (every patient, filterable) — the same table, same
rules, so the two pages can't drift apart. Rendering only: every validation
and write goes through engine/corrections.py.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import corrections as C
from engine.schemas import CorrectionField, CorrectionSource, CorrectionStatus


def _display_value(row: pd.Series) -> float | None:
    """The achieved value in its natural display unit (percent for
    VolumeAtDose, since it's stored as a 0-1 fraction) — None when it's
    genuinely missing (the case this editor exists to fix)."""
    if pd.isna(row["achieved_value"]):
        return None
    unit = C.unit_for_goal_type(row["goal_type"])
    return row["achieved_value"] * 100.0 if unit == "%" else row["achieved_value"]


def render_correction_editor(engine, corrected_df: pd.DataFrame, *, key: str) -> None:
    """corrected_df must be the output of engine.corrections.apply_corrections
    (has is_pending_review, goal_id, ...). Shows nothing but a success
    message if there's nothing pending in it."""
    pending = corrected_df[corrected_df["is_pending_review"]].copy()
    if pending.empty:
        st.success("Nothing pending here — every goal has an achieved value and a priority.")
        return

    n = len(pending)
    st.warning(f"⚠️ {n} goal{'s' if n != 1 else ''} need review before results based on "
              "them are complete.", icon="⚠️")

    pending["_missing_value"] = pending["achieved_value"].isna()
    pending["_missing_priority"] = pending["priority"].isna()
    pending["issue"] = [
        " & ".join(filter(None, [
            "missing value" if mv else None,
            "missing priority" if mp else None,
        ]))
        for mv, mp in zip(pending["_missing_value"], pending["_missing_priority"])
    ]
    pending["unit"] = pending["goal_type"].map(C.unit_for_goal_type)
    pending["value"] = pending.apply(_display_value, axis=1)
    pending["confirmed_not_evaluable"] = False
    pending["source"] = None
    pending["reason"] = ""

    corrected_by = st.text_input(
        "Your name (required to save any correction below)", key=f"{key}_corrected_by"
    )

    editable_cols = ["pt_no", "plan_type", "roi", "goal_text", "issue", "unit",
                     "value", "priority", "confirmed_not_evaluable", "source", "reason", "goal_id"]
    edited = st.data_editor(
        pending[editable_cols],
        key=f"{key}_editor",
        hide_index=True,
        disabled=["pt_no", "plan_type", "roi", "goal_text", "issue", "unit"],
        column_config={
            "pt_no": "Patient",
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
            "goal_id": None,  # kept for saving, hidden from view
        },
    )

    if not st.button("Save corrections", key=f"{key}_save"):
        return

    if not corrected_by.strip():
        st.error("Enter your name before saving.")
        return

    saved, warnings, errors = 0, [], []
    for _, edited_row in edited.iterrows():
        goal_id = int(edited_row["goal_id"])
        original_row = pending.loc[pending["goal_id"] == goal_id].iloc[0]
        label = f"{edited_row['pt_no']} / {edited_row['plan_type']} / {edited_row['roi']}"

        wants_value_correction = original_row["_missing_value"] and (
            edited_row["confirmed_not_evaluable"] or pd.notna(edited_row["value"])
        )
        if wants_value_correction:
            source = CorrectionSource(edited_row["source"]) if edited_row["source"] else None
            try:
                if edited_row["confirmed_not_evaluable"]:
                    _, warning = C.record_correction(
                        engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CONFIRMED_NOT_EVALUABLE, source=source,
                        reason=edited_row["reason"], corrected_by=corrected_by,
                    )
                else:
                    _, warning = C.record_correction(
                        engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=source,
                        reason=edited_row["reason"], corrected_by=corrected_by,
                        corrected_value_display=float(edited_row["value"]),
                    )
                saved += 1
                if warning:
                    warnings.append(f"{label}: {warning}")
            except C.CorrectionValidationError as exc:
                errors.append(f"{label}: {exc}")

        wants_priority_correction = original_row["_missing_priority"] and pd.notna(edited_row["priority"])
        if wants_priority_correction:
            source = CorrectionSource(edited_row["source"]) if edited_row["source"] else None
            try:
                C.record_correction(
                    engine, goal_id=goal_id, field=CorrectionField.PRIORITY,
                    status=CorrectionStatus.CORRECTED, source=source,
                    reason=edited_row["reason"], corrected_by=corrected_by,
                    corrected_value_display=edited_row["priority"],
                )
                saved += 1
            except C.CorrectionValidationError as exc:
                errors.append(f"{label}: {exc}")

    for w in warnings:
        st.warning(w)
    for e in errors:
        st.error(e)
    if saved and not errors:
        st.success(f"Saved {saved} correction(s).")
    if saved:
        st.rerun()
