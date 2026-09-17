"""Registry data, HN masking, and the Module 9 multi-sheet Excel export.

load_registry_frame() is the one query behind both Module 1 (Patient Data —
on screen, optionally unmasked behind a checkbox) and Module 9 (Registry
Export — always masked, for a file leaving the app). Unlike
engine.storage.load_analysis_frame, it deliberately DOES include hn: callers
decide whether/how to show it. mask_hn() is that masking.

CLAUDE.md rule 4 says HN is only ever shown masked in the registry; Module
1's unmask-behind-a-checkbox is the one authorized exception to "masked"
(never a default, always an explicit click, never in an export, a chart, or
any of the analysis tables in Modules 2-5) — see that rule's own wording.
build_analysis_workbook() is Module 9's export itself: it never offers an
unmask option, so M1_Patients is always masked.

build_analysis_workbook() re-derives the same corrected + straight-pass-
imputed pipeline frame every other compute_* function expects
(engine.storage.load_analysis_frame -> engine.corrections.apply_corrections
-> engine.imputation.apply_straight_pass) and hands it to each module's own
compute_* function — it duplicates no calculation, only assembles their
outputs into one workbook. Every call writes one engine.storage.
analysis_runs row via record_analysis_run(), so every exported file is
traceable to exactly the engine/criteria version and settings that produced
it, whether or not the caller remembered to log anything itself.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.engine import Engine
from sqlmodel import Session, select
import pandas as pd

from engine import analysis as A
from engine import corrections as C
from engine import imputation as I
from engine.corrections import pending_count_by_patient
from engine.criteria_v0 import CRITERIA_SHA256, CRITERIA_VERSION, ENGINE_VERSION
from engine.priority_filter import PrioritySelection, cohort_summary, paired_consistency, quality_index_table
from engine.schemas import PlanType
from engine.storage import Patient, Plan, load_analysis_frame, record_analysis_run

__all__ = ["mask_hn", "load_registry_frame", "build_analysis_workbook"]

_PLAN_ORDER = [p.value for p in PlanType]

# compute_pass_rate()'s own defaults (docs/analysis_manual_th_v2.md §2.4) —
# what the export uses unless a caller (Module 9's page) overrides them to
# match whatever the user last set on the Pass Rate page.
DEFAULT_PASS_RATE_SETTINGS = dict(
    exclude_non_evaluable=True, exclude_no_priority=True, matched_goals_only=False,
)


def mask_hn(hn: str) -> str:
    """'54687146' -> '****7146' (last 4 digits kept). Anything 4 characters
    or shorter is fully masked."""
    text = str(hn)
    if len(text) <= 4:
        return "*" * len(text)
    return "*" * (len(text) - 4) + text[-4:]


def load_registry_frame(engine: Engine) -> pd.DataFrame:
    """One row per patient: case metadata, which plans have been uploaded,
    whether the Auto+Manual plan was a straight pass, how many of the
    patient's goals are still pending review, whether the patient's own
    profile is complete, and when the case was created. `hn` is included
    as-is (not masked) — apply mask_hn() where it's actually displayed/
    exported.

    profile_complete is computed here, not stored: False if dose_regimen,
    tx_room, mp1, mp2 or ro is null, or if any of the patient's *uploaded*
    plans has no planning_time_min. A case can be saved with only pt_no
    and plan files (engine.schemas.FormInput) — this is what tells Module
    1 which patients still need their details filled in via its Edit
    dialog. Independent of pending_count, which is about goal-level data
    (engine.corrections), not this case-level metadata.
    """
    with Session(engine) as session:
        patients = session.exec(select(Patient)).all()
        plans = session.exec(select(Plan)).all()

    plans_by_patient: dict[int, list[Plan]] = {}
    for plan in plans:
        plans_by_patient.setdefault(plan.patient_id, []).append(plan)

    pending_by_pt_no = pending_count_by_patient(engine)

    rows = []
    for patient in patients:
        patient_plans = plans_by_patient.get(patient.id, [])
        plan_types_present = sorted(
            {p.plan_type.value for p in patient_plans}, key=_PLAN_ORDER.index
        )
        am_plan = next((p for p in patient_plans if p.plan_type == PlanType.AUTO_MANUAL), None)

        missing_plan_time = any(p.planning_time_min is None for p in patient_plans)
        profile_complete = not (
            patient.dose_regimen is None
            or patient.tx_room is None
            or patient.mp1 is None
            or patient.mp2 is None
            or patient.ro is None
            or missing_plan_time
        )

        rows.append(dict(
            pt_no=patient.pt_no,
            hn=patient.hn,
            dose_regimen=patient.dose_regimen.value if patient.dose_regimen is not None else None,
            sib_boost=patient.sib_boost,
            tx_room=patient.tx_room,
            mp1=patient.mp1,
            mp2=patient.mp2,
            ro=patient.ro,
            plans_uploaded=", ".join(plan_types_present) if plan_types_present else "—",
            straight_pass=am_plan.is_straight_pass if am_plan else None,
            pending_count=pending_by_pt_no.get(patient.pt_no, 0),
            profile_complete=profile_complete,
            created_at=patient.created_at,
        ))

    columns = ["pt_no", "hn", "dose_regimen", "sib_boost", "tx_room", "mp1", "mp2", "ro",
              "plans_uploaded", "straight_pass", "pending_count", "profile_complete", "created_at"]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# build_analysis_workbook — Module 9's one-workbook-per-run export
# --------------------------------------------------------------------------- #

_SHEET_GAP = 3  # blank rows between two tables stacked on one sheet


def _write_stacked(writer, sheet_name: str, top: pd.DataFrame, bottom: pd.DataFrame) -> None:
    top.to_excel(writer, sheet_name=sheet_name, index=False)
    bottom.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(top) + _SHEET_GAP)


def build_analysis_workbook(
    engine: Engine,
    *,
    priority_selection: Optional[PrioritySelection] = None,
    pass_rate_settings: Optional[dict] = None,
    entered_by: Optional[str] = None,
) -> bytes:
    """Build one .xlsx workbook (as bytes) covering every module, and log
    one analysis_runs row for it. Sheets, in order:

        RunInfo, M1_Patients, M2_PassRate, M2_Detail, M3_Time,
        M4_QualitySummary, M4_GoalScores, M4_Consistency,
        M5_DVHConsistency, Alerts, DataCorrections

    `priority_selection` drives the M4 sheets exactly like Module 4's own
    filter (default: PrioritySelection.all() — "All Priorities"); pass the
    same selection the user last chose on that page so the export matches
    what they were looking at. `pass_rate_settings` overrides
    DEFAULT_PASS_RATE_SETTINGS the same way — a dict with any of
    exclude_non_evaluable/exclude_no_priority/matched_goals_only.
    `entered_by` is the app session's "Entered by" name (see engine.auth /
    app.py) — recorded on the analysis_runs row and shown in RunInfo, None
    outside the app.
    """
    selection = priority_selection or PrioritySelection.all()
    settings = dict(DEFAULT_PASS_RATE_SETTINGS)
    if pass_rate_settings:
        settings.update(pass_rate_settings)

    raw = load_analysis_frame(engine)
    pipeline_df = I.apply_straight_pass(C.apply_corrections(raw, engine))

    registry = load_registry_frame(engine).copy()
    registry["hn"] = registry["hn"].map(mask_hn)  # M1_Patients is never unmasked in an export

    pass_rate = A.compute_pass_rate(pipeline_df, **settings)
    time_eff = A.compute_time_efficiency(pipeline_df)

    goal_scores = A.compute_goal_scores(pipeline_df)
    qi = quality_index_table(goal_scores, selection)
    qi_cohort = cohort_summary(qi)
    consistency = paired_consistency(qi)

    dvh_frame = A.compute_dvh_frame(pipeline_df)
    # One call across every unit_category: compute_dvh_consistency groups by
    # goal_key, and each goal_key belongs to exactly one unit_category, so
    # mixing units in the input doesn't mix them within a result row.
    dvh_consistency = A.compute_dvh_consistency(dvh_frame, value_col="value")

    alerts = A.compute_critical_alerts(pipeline_df)
    data_corrections = C.load_all_corrections(engine)

    settings_json = json.dumps(dict(
        pass_rate_settings=settings,
        priority_filter=selection.label,
        priority_levels=list(selection.levels),
    ))
    timestamp = datetime.now(timezone.utc)
    run_id = record_analysis_run(
        engine, engine_version=ENGINE_VERSION,
        criteria_version=CRITERIA_VERSION.upper(), settings_json=settings_json,
        entered_by=entered_by,
    )

    run_info = pd.DataFrame([
        dict(field="Engine version", value=ENGINE_VERSION),
        dict(field="Criteria version", value=CRITERIA_VERSION.upper()),
        dict(field="Criteria SHA-256", value=CRITERIA_SHA256),
        dict(field="Analysis run id", value=run_id),
        dict(field="Entered by", value=entered_by or "—"),
        dict(field="Active priority filter (M4)", value=selection.label),
        dict(field="Pass rate: exclude non-evaluable goals", value=settings["exclude_non_evaluable"]),
        dict(field="Pass rate: exclude goals without priority", value=settings["exclude_no_priority"]),
        dict(field="Pass rate: matched goals only", value=settings["matched_goals_only"]),
        dict(field="Export timestamp (UTC)", value=timestamp.isoformat(timespec="seconds")),
    ], columns=["field", "value"])

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        run_info.to_excel(writer, sheet_name="RunInfo", index=False)
        registry.to_excel(writer, sheet_name="M1_Patients", index=False)
        _write_stacked(writer, "M2_PassRate", pass_rate.per_patient, pass_rate.cohort)
        _write_stacked(writer, "M2_Detail", pass_rate.per_patient, pass_rate.priority1)
        _write_stacked(writer, "M3_Time", time_eff.per_patient, time_eff.cohort)
        _write_stacked(writer, "M4_QualitySummary", qi, qi_cohort)
        goal_scores.to_excel(writer, sheet_name="M4_GoalScores", index=False)
        consistency.to_excel(writer, sheet_name="M4_Consistency", index=False)
        dvh_consistency.to_excel(writer, sheet_name="M5_DVHConsistency", index=False)
        alerts.to_excel(writer, sheet_name="Alerts", index=False)
        data_corrections.to_excel(writer, sheet_name="DataCorrections", index=False)

    return buffer.getvalue()
