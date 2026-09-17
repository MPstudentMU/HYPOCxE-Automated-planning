"""Goal-level scoring against Criteria V0.

score_goal(row): the per-goal algorithm — classification, Improvement %,
base score, max-achievable base, multiplier, weighted score — from one row
of the (corrected, straight-pass-imputed) analysis frame. Every rule value
it applies comes from engine/criteria_v0.py (frozen); nothing here decides
what a threshold *is*, only how to apply it to a goal.

compute_goal_scores(df): score_goal applied across a whole cohort, into the
exact table engine/priority_filter.py already expects (patient, plan,
priority, category, base_score, weighted_score, max_weighted, scorable,
structure_group — see that module's own docstring), plus extra columns for
traceability (docs/analysis_manual_th_v2.md §3.5's "goal-level detail table").

compute_critical_alerts(df): §3.6 — main-PTV coverage safety check,
independent of priority filtering and of the score above.

Expected pipeline before either: engine.storage.load_analysis_frame ->
engine.corrections.apply_corrections -> engine.imputation.apply_straight_pass
-> compute_goal_scores / compute_critical_alerts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from engine import criteria_v0 as V
from engine.schemas import CriteriaDirection, StructureClass

__all__ = ["ScoredGoal", "score_goal", "compute_goal_scores", "compute_critical_alerts"]


@dataclass(frozen=True)
class ScoredGoal:
    """The full, auditable result of scoring one goal row."""

    category: str
    """CATEGORY_OAR / CATEGORY_TARGET / CATEGORY_HOTSPOT — always set, even
    for a non-scorable row (it's a property of the goal's definition, not
    of whether it happened to get measured)."""

    structure_group: Optional[str]
    """§3.7's radar-chart group; None if this roi/category isn't on the radar."""

    scorable: bool
    """False when AchievedValue is missing or Priority is missing/unset —
    §3.4. base_score/weighted_score/max_weighted are None when this is False."""

    is_zero_limit: bool
    is_coverage_ceiling: bool

    improvement_pct: Optional[float]
    """None for a zero-limit goal (the formula isn't used — Goal = 0 makes
    it undefined) or a non-scorable goal."""

    base_score: Optional[float]
    max_base: Optional[float]
    multiplier: Optional[float]
    weighted_score: Optional[float]
    max_weighted: Optional[float]


def _is_zero_limit(criteria: str, acceptance_level: float) -> bool:
    return criteria == CriteriaDirection.AT_MOST.value and acceptance_level == V.ZERO_LIMIT_ACCEPTANCE_LEVEL


def _is_coverage_ceiling(criteria: str, goal_type: str, acceptance_level: float) -> bool:
    return (criteria == CriteriaDirection.AT_LEAST.value
            and goal_type == V.COVERAGE_CEILING_GOAL_TYPE
            and acceptance_level >= V.COVERAGE_CEILING_ACCEPTANCE_LEVEL)


def _classify_category(criteria: str, roi: str, structure_class: str) -> str:
    """§3.1's classification table."""
    if criteria == CriteriaDirection.AT_LEAST.value:
        return V.CATEGORY_TARGET
    # AtMost from here on.
    if structure_class == StructureClass.TARGET.value or roi == "BODY":
        return V.CATEGORY_HOTSPOT
    return V.CATEGORY_OAR


def _structure_group(category: str, roi: str) -> Optional[str]:
    if category == V.CATEGORY_TARGET:
        return V.TARGET_STRUCTURE_GROUP
    return V.STRUCTURE_GROUPS.get(roi)


def score_goal(row) -> ScoredGoal:
    """row must expose (as attributes — a pandas Series or itertuples()
    row both work): roi, criteria, goal_type, acceptance_level,
    achieved_value, priority, evaluable, structure_class.
    """
    category = _classify_category(row.criteria, row.roi, row.structure_class)
    structure_group = _structure_group(category, row.roi)

    scorable = bool(row.evaluable) and pd.notna(row.priority)
    if not scorable:
        return ScoredGoal(
            category=category, structure_group=structure_group, scorable=False,
            is_zero_limit=False, is_coverage_ceiling=False, improvement_pct=None,
            base_score=None, max_base=None, multiplier=None,
            weighted_score=None, max_weighted=None,
        )

    zero_limit = _is_zero_limit(row.criteria, row.acceptance_level)
    coverage_ceiling = _is_coverage_ceiling(row.criteria, row.goal_type, row.acceptance_level)

    multiplier = (V.TARGET_MULTIPLIER if category == V.CATEGORY_TARGET
                 else V.OAR_HOTSPOT_MULTIPLIER)[int(row.priority)]

    if zero_limit:
        improvement_pct = None
        base_score = (V.ZERO_LIMIT_BASE_SCORE_MET if row.achieved_value == 0.0
                      else V.ZERO_LIMIT_BASE_SCORE_NOT_MET)
    else:
        if row.acceptance_level == 0:
            # Genuinely outside the specified rules (§3.4 only defines the
            # zero-target case for AtMost) — refuse to guess rather than
            # divide by zero or silently fabricate a score.
            raise ValueError(
                f"acceptance_level == 0 with criteria={row.criteria!r} (roi={row.roi!r}) "
                "isn't covered by the zero-limit rule (AtMost only) — can't compute Improvement %."
            )
        if row.criteria == CriteriaDirection.AT_MOST.value:
            improvement_pct = (row.acceptance_level - row.achieved_value) / row.acceptance_level * 100.0
        else:
            improvement_pct = (row.achieved_value - row.acceptance_level) / row.acceptance_level * 100.0

        if category == V.CATEGORY_TARGET:
            base_score = V.band_base_score(improvement_pct, V.TARGET_SCALE)
        else:
            oar_base = V.band_base_score(improvement_pct, V.OAR_SCALE)
            base_score = min(oar_base, V.HOTSPOT_MAX_SCORE) if category == V.CATEGORY_HOTSPOT else oar_base

    if zero_limit:
        max_base = V.ZERO_LIMIT_MAX_BASE
    elif coverage_ceiling:
        max_base = V.COVERAGE_CEILING_MAX_BASE
    elif category == V.CATEGORY_HOTSPOT:
        max_base = V.HOTSPOT_MAX_SCORE
    else:
        max_base = V.DEFAULT_MAX_BASE

    return ScoredGoal(
        category=category, structure_group=structure_group, scorable=True,
        is_zero_limit=zero_limit, is_coverage_ceiling=coverage_ceiling,
        improvement_pct=improvement_pct, base_score=base_score, max_base=max_base,
        multiplier=multiplier, weighted_score=base_score * multiplier,
        max_weighted=max_base * multiplier,
    )


def compute_goal_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per goal per plan per patient. df is the (corrected,
    straight-pass-imputed) analysis frame. A row that genuinely can't be
    scored (§3.4's "not covered" edge case in score_goal) is kept with
    scorable=False and a note, rather than crashing the whole cohort's
    computation over one bad goal.
    """
    columns = [
        "patient", "plan", "priority", "category", "base_score", "weighted_score",
        "max_weighted", "scorable", "structure_group",
        # traceability (§3.5's "goal-level scoring detail" table)
        "goal_id", "goal_key", "roi", "goal_text", "goal_type", "criteria",
        "acceptance_level", "achieved_value", "improvement_pct", "max_base",
        "multiplier", "is_zero_limit", "is_coverage_ceiling", "error",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for row in df.itertuples():
        try:
            scored = score_goal(row)
            error = None
        except ValueError as exc:
            scored = ScoredGoal(
                category=_classify_category(row.criteria, row.roi, row.structure_class),
                structure_group=None, scorable=False, is_zero_limit=False,
                is_coverage_ceiling=False, improvement_pct=None, base_score=None,
                max_base=None, multiplier=None, weighted_score=None, max_weighted=None,
            )
            error = str(exc)

        rows.append(dict(
            patient=row.pt_no, plan=row.plan_type, priority=row.priority, category=scored.category,
            base_score=scored.base_score, weighted_score=scored.weighted_score,
            max_weighted=scored.max_weighted, scorable=scored.scorable,
            structure_group=scored.structure_group,
            goal_id=row.goal_id, goal_key=row.goal_key, roi=row.roi, goal_text=row.goal_text,
            goal_type=row.goal_type, criteria=row.criteria, acceptance_level=row.acceptance_level,
            achieved_value=row.achieved_value, improvement_pct=scored.improvement_pct,
            max_base=scored.max_base, multiplier=scored.multiplier,
            is_zero_limit=scored.is_zero_limit, is_coverage_ceiling=scored.is_coverage_ceiling,
            error=error,
        ))

    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# compute_critical_alerts — §3.6
# --------------------------------------------------------------------------- #


def compute_critical_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (patient, plan): main-PTV coverage vs the 95%-of-Rx
    safety threshold. Independent of priority filtering and of scoring —
    see the module docstring. df is the (corrected, straight-pass-imputed)
    analysis frame; must carry rx_cgy (engine.storage.load_analysis_frame
    already does).
    """
    columns = ["patient", "plan", "status", "coverage", "basis", "rx_cgy", "target_dose_cgy"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (patient, plan), group in df.groupby(["pt_no", "plan_type"], sort=False):
        rx_cgy = group["rx_cgy"].iloc[0] if "rx_cgy" in group.columns else None
        ptv = group[group["roi"] == V.CRITICAL_ALERT_ROI]

        matched = ptv.iloc[0:0]
        basis = None
        target_dose_cgy = None

        if pd.notna(rx_cgy):
            target_dose_cgy = V.CRITICAL_ALERT_COVERAGE_FRACTION * rx_cgy
            tolerance = V.CRITICAL_ALERT_DOSE_TOLERANCE_FRACTION * rx_cgy
            primary = ptv[
                (ptv["goal_type"] == V.CRITICAL_ALERT_GOAL_TYPE)
                & (ptv["criteria"] == CriteriaDirection.AT_LEAST.value)
                & ((ptv["parameter_value"] - target_dose_cgy).abs() <= tolerance)
            ]
            if not primary.empty:
                matched = primary

        if matched.empty:
            fallback = ptv[
                (ptv["goal_type"] == V.CRITICAL_ALERT_GOAL_TYPE)
                & (ptv["criteria"] == CriteriaDirection.AT_LEAST.value)
                & ((ptv["acceptance_level"] - V.CRITICAL_ALERT_COVERAGE_FRACTION).abs() < 1e-6)
            ]
            if not fallback.empty:
                matched = fallback
                target_dose_cgy = float(fallback["parameter_value"].iloc[0])
                pct_rx = (target_dose_cgy / rx_cgy * 100.0) if pd.notna(rx_cgy) and rx_cgy else None
                basis = f"fallback: V{pct_rx:.0f}%Rx" if pct_rx is not None else "fallback"

        evaluable = matched[matched["achieved_value"].notna()]
        if evaluable.empty:
            rows.append(dict(patient=patient, plan=plan, status="No PTV coverage goal found",
                             coverage=None, basis=basis, rx_cgy=rx_cgy,
                             target_dose_cgy=target_dose_cgy if not matched.empty else None))
            continue

        coverage = float(evaluable["achieved_value"].min())
        status = "FAIL (Unacceptable)" if coverage < V.CRITICAL_ALERT_COVERAGE_FRACTION else "OK"
        rows.append(dict(patient=patient, plan=plan, status=status, coverage=coverage,
                         basis=basis, rx_cgy=rx_cgy, target_dose_cgy=target_dose_cgy))

    return pd.DataFrame(rows, columns=columns)
