"""Manual data corrections and their audit trail.

Some goal_results rows come out of engine/parser.py with no AchievedValue
(the plan export had nothing to give) or no Priority (RayStation's
2147483647 "no priority" sentinel, normalized to None). Those rows are kept,
not dropped, precisely so a human can review and fix them here — see
pages/0_new_case.py (upload preview) and pages/8_data_review.py (everything
still pending, across patients).

A correction is always an overlay, never an edit: goal_corrections records
what changed, why, by whom, and from what source, but engine/storage.py's
goal_results is never written to by this module. apply_corrections(df)
takes the raw analysis frame (engine.storage.load_analysis_frame) and
returns a corrected copy — recomputing Status for any row whose
AchievedValue changed, and tagging every affected row "Corrected" (or
"Confirmed not evaluable"). It must run before straight-pass imputation
(engine/imputation.py, Phase 2c): imputation decides whether to copy a
plan's rows into another plan, and it should copy the *corrected* values,
not the raw export's gaps — see copy_corrections_to_new_goal below for the
matching half of that (making sure a correction travels with a row that
imputation copies).

Nothing here computes a metric; apply_corrections just puts the right
values in front of whatever does (pass rate, scoring — Phases 3+).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from engine.parser import compute_status
from engine.schemas import (
    VALID_PRIORITIES,
    CorrectionField,
    CorrectionSource,
    CorrectionStatus,
    CriteriaDirection,
)
from engine.storage import GoalCorrection, GoalResult, Patient, Plan, load_analysis_frame

__all__ = [
    "CorrectionValidationError",
    "GOAL_TYPE_UNITS",
    "DEFAULT_UNIT",
    "PERCENT_GOAL_TYPES",
    "unit_for_goal_type",
    "to_storage_value",
    "validate_value_range",
    "is_far_from_goal",
    "record_correction",
    "apply_corrections",
    "copy_corrections_to_new_goal",
    "pending_count",
    "pending_count_by_patient",
    "load_all_corrections",
]


class CorrectionValidationError(ValueError):
    """A correction can't be saved as given (missing source/reason, an
    out-of-range value, an invalid priority, ...). Pages should catch this
    and show it with st.error rather than letting it crash the app."""


# --------------------------------------------------------------------------- #
# GoalType -> unit
# --------------------------------------------------------------------------- #

# The unit an AchievedValue is expressed in, keyed by GoalType. RayStation
# stores a VolumeAtDose goal's AcceptanceLevel/AchievedValue as a *fraction*
# (0-1) even though the goal text reads "V2000cGy <= 65.0%" — so a value a
# person types as a percentage has to be divided by 100 before it's stored,
# to stay consistent with every other row of that GoalType already in the
# database. No other GoalType observed in pilot data needs this conversion.
GOAL_TYPE_UNITS: dict[str, str] = {
    "DoseAtVolume": "cGy",
    "DoseAtAbsoluteVolume": "cGy",
    "AverageDose": "cGy",
    "AbsoluteVolumeAtDose": "cc",
    "VolumeAtDose": "%",
}
DEFAULT_UNIT = "cGy"  # a GoalType we haven't seen yet: dose is the common case
PERCENT_GOAL_TYPES = frozenset(gt for gt, unit in GOAL_TYPE_UNITS.items() if unit == "%")


def unit_for_goal_type(goal_type: str) -> str:
    """The unit label to show next to the editable value cell."""
    return GOAL_TYPE_UNITS.get(goal_type, DEFAULT_UNIT)


def to_storage_value(value_display: float, goal_type: str) -> float:
    """Convert a value as a person typed it (in unit_for_goal_type's unit)
    to the unit goal_results actually stores it in. Only VolumeAtDose needs
    this (percent -> fraction); everything else is already stored in its
    display unit and passes through unchanged."""
    if goal_type in PERCENT_GOAL_TYPES:
        return value_display / 100.0
    return value_display


def validate_value_range(value_display: float, goal_type: str) -> None:
    """Raise CorrectionValidationError if value_display (in its display
    unit) is out of range: 0-100 for a percent GoalType, >=0 for dose/cc."""
    unit = unit_for_goal_type(goal_type)
    if unit == "%":
        if not (0 <= value_display <= 100):
            raise CorrectionValidationError(
                f"{value_display:g}% is out of range — a percentage must be between 0 and 100."
            )
    else:
        if value_display < 0:
            raise CorrectionValidationError(
                f"{value_display:g} {unit} is out of range — a dose or volume can't be negative."
            )


def is_far_from_goal(stored_value: float, acceptance_level: float, threshold: float = 0.5) -> bool:
    """True if stored_value (in storage units, i.e. already converted) is
    more than `threshold` (50% by default) away from acceptance_level in
    relative terms. Used only to warn — the value is still allowed to save."""
    if acceptance_level == 0:
        return abs(stored_value) > 1e-9
    return abs(stored_value - acceptance_level) / abs(acceptance_level) > threshold


# --------------------------------------------------------------------------- #
# Writing a correction
# --------------------------------------------------------------------------- #


def record_correction(
    engine: Engine,
    *,
    goal_id: int,
    field: CorrectionField,
    status: CorrectionStatus,
    source: CorrectionSource,
    reason: str,
    corrected_by: str,
    corrected_value_display: Optional[float] = None,
) -> tuple[int, Optional[str]]:
    """Validate and insert one goal_corrections row. Never touches
    goal_results. Returns (correction_id, warning_or_None) — the warning is
    set (but the correction still saved) when the value is more than 50%
    away from the goal.

    Raises CorrectionValidationError for anything that shouldn't save at
    all: no source, no reason, a missing value with status CORRECTED, an
    out-of-range value, or CONFIRMED_NOT_EVALUABLE on a Priority field
    (a missing priority always needs an actual value).
    """
    if source is None:
        raise CorrectionValidationError("Select a source before saving.")
    if not reason or not reason.strip():
        raise CorrectionValidationError("Enter a reason before saving.")
    if not corrected_by or not corrected_by.strip():
        raise CorrectionValidationError("Enter who is making this correction.")

    if field == CorrectionField.PRIORITY and status == CorrectionStatus.CONFIRMED_NOT_EVALUABLE:
        raise CorrectionValidationError(
            "A missing priority needs an actual priority (1/2/3) — "
            "'Confirmed not evaluable' only applies to a missing achieved value."
        )

    with Session(engine) as session:
        goal = session.get(GoalResult, goal_id)
        if goal is None:
            raise CorrectionValidationError(f"No such goal (id={goal_id}).")

        warning: Optional[str] = None
        unit_entered: Optional[str] = None

        if field == CorrectionField.ACHIEVED_VALUE:
            original_value = None if goal.achieved_value is None else str(goal.achieved_value)
            if status == CorrectionStatus.CORRECTED:
                if corrected_value_display is None:
                    raise CorrectionValidationError(
                        "Enter a value, or choose 'Confirmed not evaluable'."
                    )
                validate_value_range(corrected_value_display, goal.goal_type)
                unit_entered = unit_for_goal_type(goal.goal_type)
                stored_value = to_storage_value(corrected_value_display, goal.goal_type)
                if is_far_from_goal(stored_value, goal.acceptance_level):
                    warning = (
                        f"{corrected_value_display:g}{unit_entered} is more than 50% away from "
                        f"the goal ({goal.acceptance_level:g}) — saved anyway; please double-check."
                    )
                corrected_value = str(stored_value)
            else:
                corrected_value = None
        elif field == CorrectionField.PRIORITY:
            original_value = None if goal.priority is None else str(goal.priority)
            if corrected_value_display is None or int(corrected_value_display) not in VALID_PRIORITIES:
                raise CorrectionValidationError(f"Priority must be one of {VALID_PRIORITIES}.")
            corrected_value = str(int(corrected_value_display))
        else:
            raise CorrectionValidationError(f"Unknown field {field!r}.")

        correction = GoalCorrection(
            goal_id=goal_id,
            field=field,
            original_value=original_value,
            corrected_value=corrected_value,
            unit_entered=unit_entered,
            status=status,
            source=source,
            reason=reason.strip(),
            corrected_by=corrected_by.strip(),
        )
        session.add(correction)
        session.commit()
        session.refresh(correction)
        return correction.id, warning


# --------------------------------------------------------------------------- #
# Applying corrections at analysis time
# --------------------------------------------------------------------------- #


def apply_corrections(df: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    """Overlay active (non-superseded) goal_corrections onto the analysis
    frame from engine.storage.load_analysis_frame (which must carry a
    goal_id column). Never modifies the database; always returns a fresh
    corrected copy computed from the raw row + the corrections table, so
    there is exactly one place any of this logic lives.

    Adds three columns:
      correction_tag           "Corrected" / "Confirmed not evaluable" / None
      confirmed_not_evaluable  bool
      is_pending_review        bool — True while the goal still needs
                                attention: AchievedValue missing and not
                                confirmed not-evaluable, or Priority missing.
                                Scoring/pass rate (Phases 3+) should filter
                                these out until resolved.

    Run this before straight-pass imputation (engine/imputation.py).
    """
    df = df.copy()
    df["correction_tag"] = None
    df["confirmed_not_evaluable"] = False

    if not df.empty and "goal_id" in df.columns:
        goal_ids = [int(g) for g in df["goal_id"].dropna().unique().tolist()]
        if goal_ids:
            with Session(engine) as session:
                corrections = session.exec(
                    select(GoalCorrection)
                    .where(GoalCorrection.goal_id.in_(goal_ids))
                    .where(GoalCorrection.superseded_by_upload == False)  # noqa: E712
                ).all()

            # last write wins if a goal was corrected more than once
            latest: dict[tuple[int, CorrectionField], GoalCorrection] = {}
            for c in corrections:
                key = (c.goal_id, c.field)
                if key not in latest or c.corrected_at >= latest[key].corrected_at:
                    latest[key] = c

            for (goal_id, corr_field), correction in latest.items():
                mask = df["goal_id"] == goal_id
                if not mask.any():
                    continue

                if corr_field == CorrectionField.ACHIEVED_VALUE:
                    if correction.status == CorrectionStatus.CORRECTED:
                        new_value = float(correction.corrected_value)
                        df.loc[mask, "achieved_value"] = new_value
                        df.loc[mask, "evaluable"] = True
                        df.loc[mask, "correction_tag"] = "Corrected"
                        for i in df.index[mask]:
                            criteria = CriteriaDirection(df.at[i, "criteria"])
                            new_status = compute_status(new_value, df.at[i, "acceptance_level"], criteria)
                            df.at[i, "status"] = new_status.value if new_status else None
                    else:  # CONFIRMED_NOT_EVALUABLE
                        df.loc[mask, "confirmed_not_evaluable"] = True
                        df.loc[mask, "correction_tag"] = "Confirmed not evaluable"
                        # A raw export can carry a stale Status (RayStation
                        # itself sometimes writes "FAIL" as a conservative
                        # default when AchievedValue is blank) — misleading
                        # once someone has confirmed the goal genuinely
                        # can't be evaluated, so clear it rather than show a
                        # false failure.
                        df.loc[mask, "status"] = None

                elif corr_field == CorrectionField.PRIORITY:
                    df.loc[mask, "priority"] = int(correction.corrected_value)
                    still_untagged = mask & df["correction_tag"].isna()
                    df.loc[still_untagged, "correction_tag"] = "Corrected"

    # Guarantee, for every row this function returns (corrected or not):
    # Status is set only when it's trustworthy. A raw export can carry a
    # stale Status (RayStation itself sometimes writes "FAIL" even when
    # AchievedValue is blank) — clear it whenever AchievedValue is still
    # missing and not confirmed not-evaluable, so any future scoring/pass-
    # rate code can safely filter on `status.notna()` alone rather than
    # every caller having to also remember to check is_pending_review.
    still_unresolved_achieved = df["achieved_value"].isna() & ~df["confirmed_not_evaluable"]
    df.loc[still_unresolved_achieved, "status"] = None

    df["is_pending_review"] = still_unresolved_achieved | df["priority"].isna()
    return df


# --------------------------------------------------------------------------- #
# Straight-pass support: carry a correction along with a copied goal
# --------------------------------------------------------------------------- #


def pending_count(engine: Engine) -> int:
    """How many goal_results rows, across every patient, still need review
    (missing AchievedValue and not confirmed not-evaluable, or missing
    Priority). Backs the pending-count banner shown on every module page.
    """
    df = load_analysis_frame(engine)
    if df.empty:
        return 0
    return int(apply_corrections(df, engine)["is_pending_review"].sum())


def pending_count_by_patient(engine: Engine) -> dict[str, int]:
    """{pt_no: n_pending} across the whole cohort — backs Module 1's alert
    status column."""
    df = load_analysis_frame(engine)
    if df.empty:
        return {}
    corrected = apply_corrections(df, engine)
    return corrected.groupby("pt_no")["is_pending_review"].sum().astype(int).to_dict()


def copy_corrections_to_new_goal(engine: Engine, *, source_goal_id: int, target_goal_id: int) -> int:
    """Copy every active correction on source_goal_id onto target_goal_id.

    For when a goal_results row is duplicated across plans — e.g.
    engine/imputation.py's straight-pass rule copying an Auto plan's rows
    into a missing/blank Auto+Manual plan — so a correction made on the
    original goal stays attached to the copy instead of silently applying
    to only one of the two plans. Returns how many corrections were copied.
    """
    with Session(engine) as session:
        source_corrections = session.exec(
            select(GoalCorrection)
            .where(GoalCorrection.goal_id == source_goal_id)
            .where(GoalCorrection.superseded_by_upload == False)  # noqa: E712
        ).all()
        for c in source_corrections:
            session.add(GoalCorrection(
                goal_id=target_goal_id,
                field=c.field,
                original_value=c.original_value,
                corrected_value=c.corrected_value,
                unit_entered=c.unit_entered,
                status=c.status,
                source=c.source,
                reason=f"{c.reason} (copied by straight-pass from goal {source_goal_id})",
                corrected_by=c.corrected_by,
            ))
        session.commit()
        return len(source_corrections)


def load_all_corrections(engine: Engine) -> pd.DataFrame:
    """Every goal_corrections row ever written, with enough context to be
    readable on its own — patient (pt_no, never hn), plan, roi, goal_text
    — for Module 9's DataCorrections export sheet and any other full
    audit-trail view. Superseded corrections are included (flagged
    superseded_by_upload=True) — an audit trail's job is completeness,
    not just what's currently applied; see apply_corrections for that.

    The joins to GoalResult/Plan/Patient are LEFT OUTER, not inner: a
    re-upload (engine.storage.replace_plan) deletes the old goal_results
    row a superseded correction targeted (after flagging it
    superseded_by_upload — see _upsert_plan), so goal_id can point to a
    row that no longer exists. An inner join would silently drop that
    correction from the audit trail — exactly the history this function
    exists to keep. pt_no/plan_type/roi/goal_text come back None for such
    a row; the correction itself, and why it happened, doesn't.
    """
    columns = ["id", "field", "original_value", "corrected_value", "unit_entered",
              "status", "source", "reason", "corrected_by", "corrected_at",
              "superseded_by_upload", "pt_no", "plan_type", "roi", "goal_text"]
    with Session(engine) as session:
        rows = session.exec(
            select(
                GoalCorrection.id, GoalCorrection.field, GoalCorrection.original_value,
                GoalCorrection.corrected_value, GoalCorrection.unit_entered,
                GoalCorrection.status, GoalCorrection.source, GoalCorrection.reason,
                GoalCorrection.corrected_by, GoalCorrection.corrected_at,
                GoalCorrection.superseded_by_upload,
                Patient.pt_no, Plan.plan_type, GoalResult.roi, GoalResult.goal_text,
            )
            .select_from(GoalCorrection)
            .join(GoalResult, GoalResult.id == GoalCorrection.goal_id, isouter=True)
            .join(Plan, Plan.id == GoalResult.plan_id, isouter=True)
            .join(Patient, Patient.id == Plan.patient_id, isouter=True)
            .order_by(GoalCorrection.corrected_at.desc())
        ).all()

    df = pd.DataFrame(rows, columns=columns)
    for col in ("field", "status", "source", "plan_type"):
        df[col] = df[col].map(lambda v: v.value if isinstance(v, Enum) else v)
    assert "hn" not in df.columns, "HN must never enter the corrections audit trail — CLAUDE.md rule 4"
    return df
