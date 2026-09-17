"""Handling of missing values: the Straight-Pass Exception Rule.

docs/analysis_manual_th_v2.md §2.5 — when a patient's Auto plan passed
without needing manual touch-up, no Auto+Manual plan is ever uploaded (or
one exists with every AchievedValue blank). apply_straight_pass() copies
that patient's Auto goal rows (and planning time, since they travel
together on the same row) into Auto+Manual, so cohort statistics for
Auto+Manual are computed from the full sample rather than only the
patients who needed a real manual touch-up.

This is an analysis-time-only transform, same as engine.corrections: the
manual is explicit that raw data is never modified ("ข้อมูลดิบในฐานข้อมูล
ไม่ถูกแก้ไข การเติมข้อมูลเกิดขึ้น ณ เวลาวิเคราะห์") — nothing here writes to
engine/storage.py. If a real Auto+Manual plan is uploaded later, it simply
stops qualifying for this rule and its own data is used instead.

Pipeline: engine.storage.load_analysis_frame -> engine.corrections
.apply_corrections -> engine.imputation.apply_straight_pass -> scoring /
engine.analysis.compute_pass_rate.
"""
from __future__ import annotations

import pandas as pd

from engine.schemas import PlanType

__all__ = ["needs_straight_pass", "apply_straight_pass"]


def needs_straight_pass(patient_df: pd.DataFrame) -> bool:
    """True if this patient's rows (all plans) call for the straight-pass
    rule: has an Auto plan, and Auto+Manual is either absent or entirely
    without an AchievedValue."""
    has_auto = (patient_df["plan_type"] == PlanType.AUTO.value).any()
    if not has_auto:
        return False
    auto_manual = patient_df[patient_df["plan_type"] == PlanType.AUTO_MANUAL.value]
    return auto_manual.empty or auto_manual["achieved_value"].isna().all()


def apply_straight_pass(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of df where every patient needing the straight-pass
    rule has its (absent/entirely-blank) Auto+Manual rows replaced by a
    copy of its Auto rows — same goal_id, achieved_value, status,
    planning_time_min and everything else, just plan_type swapped to
    Auto+Manual. Adds an `imputed_from_auto` column (True only for those
    copied rows) so callers/charts can flag them, per the manual's own
    "show straight-pass patients in Summary and chart hover" requirement.
    """
    df = df.copy()
    df["imputed_from_auto"] = False
    if df.empty:
        return df

    pieces = []
    for _, patient_df in df.groupby("pt_no", sort=False):
        if not needs_straight_pass(patient_df):
            pieces.append(patient_df)
            continue
        kept = patient_df[patient_df["plan_type"] != PlanType.AUTO_MANUAL.value]
        imputed = patient_df[patient_df["plan_type"] == PlanType.AUTO.value].copy()
        imputed["plan_type"] = PlanType.AUTO_MANUAL.value
        imputed["imputed_from_auto"] = True
        pieces.append(pd.concat([kept, imputed], ignore_index=True))

    return pd.concat(pieces, ignore_index=True) if pieces else df
