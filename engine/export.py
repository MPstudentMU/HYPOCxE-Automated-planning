"""Registry data and HN masking.

load_registry_frame() is the one query behind both Module 1 (Patient Data —
on screen, optionally unmasked behind a checkbox) and Module 9 (Registry
Export — always masked, for a file leaving the app). Unlike
engine.storage.load_analysis_frame, it deliberately DOES include hn: callers
decide whether/how to show it. mask_hn() is that masking.

CLAUDE.md rule 4 says HN is only ever shown masked in the registry; Module
1's unmask-behind-a-checkbox is the one authorized exception to "masked"
(never a default, always an explicit click, never in an export, a chart, or
any of the analysis tables in Modules 2-5) — see that rule's own wording.
"""
from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlmodel import Session, select
import pandas as pd

from engine.corrections import pending_count_by_patient
from engine.schemas import PlanType
from engine.storage import Patient, Plan

__all__ = ["mask_hn", "load_registry_frame"]

_PLAN_ORDER = [p.value for p in PlanType]


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
    patient's goals are still pending review, and when the case was
    created. `hn` is included as-is (not masked) — apply mask_hn() where
    it's actually displayed/exported.
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

        rows.append(dict(
            pt_no=patient.pt_no,
            hn=patient.hn,
            dose_regimen=patient.dose_regimen.value,
            sib_boost=patient.sib_boost,
            tx_room=patient.tx_room,
            mp1=patient.mp1,
            mp2=patient.mp2,
            ro=patient.ro,
            plans_uploaded=", ".join(plan_types_present) if plan_types_present else "—",
            straight_pass=am_plan.is_straight_pass if am_plan else None,
            pending_count=pending_by_pt_no.get(patient.pt_no, 0),
            created_at=patient.created_at,
        ))

    columns = ["pt_no", "hn", "dose_regimen", "sib_boost", "tx_room", "mp1", "mp2", "ro",
              "plans_uploaded", "straight_pass", "pending_count", "created_at"]
    return pd.DataFrame(rows, columns=columns)
