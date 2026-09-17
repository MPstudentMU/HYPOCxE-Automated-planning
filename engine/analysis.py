"""Cohort-level analysis built on scored goals.

compute_pass_rate() — docs/analysis_manual_th_v2.md's own Module 1
(this app's page, Module 2 "Pass Rate"):

    % Pass Rate = (goals with Status PASS / goals considered) × 100

per patient × plan. "Goals considered" excludes non-evaluable goals and
goals with no priority by default (both toggleable — §2.4's settings
table), and can optionally be further restricted to goal_keys present in
all of a patient's plans ("Matched goals only", default off). Every
excluded/non-evaluable count is reported alongside the rate — never
folded in silently.

compute_time_efficiency() — the manual's own Module 2 (this app's page,
Module 3 "Time Efficiency"):

    % Time Efficiency (plan) = (T_manual − T_plan) / T_manual × 100

for plan in {Auto, Auto+Manual}; None if either time is missing (not 0 —
a genuine "can't compute this", not a 0% claim). Banded per §Module 2's
table: <40% Below, 40-75% Target, >75% Exceeds. A straight-pass patient
(engine.imputation.apply_straight_pass already copied Auto's
planning_time_min onto Auto+Manual) gets identical times for both, so
identical efficiency, for free — no special-casing needed here.

Expected pipeline before either: engine.storage.load_analysis_frame ->
engine.corrections.apply_corrections -> engine.imputation.apply_straight_pass
-> compute_pass_rate / compute_time_efficiency.

compute_goal_scores() / compute_critical_alerts() are re-exported here from
engine.scoring, where they're actually implemented (alongside score_goal()
and Criteria V0's application — see that module's docstring for why they
live there). pages/4_plan_quality.py's ADAPTER section was written against
`engine.analysis.compute_goal_scores`/`compute_critical_alerts` before
either existed; re-exporting satisfies that without duplicating the
implementation or moving it out of what's otherwise a clean split (goal-
level scoring vs. cohort-level analysis).

compute_dvh_frame() — Module 3's DVH Consistency boxplot (§3.7): one row
per (patient, plan, goal), the achieved value converted to its natural
display unit (Dose in Gy, %Volume, or Volume in cc — never mixed on one
axis) plus an always-available Normalised (% of Goal) value. Independent
of priority filtering — it's a descriptive summary of raw DVH values, not
a scored metric.

compute_dvh_consistency() — one row per goal: n/mean/SD/CV per plan (an
n < 3 cell is marked insufficient_n rather than shown with a misleading
SD), and, for Auto and Auto+Manual each vs Manual, whether that plan's SD
is lower and the paired engine.stats.pitman_morgan_test /
Wilcoxon signed-rank results (both None below 3 paired patients).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine.schemas import GoalStatus, PlanType
from engine.scoring import compute_critical_alerts, compute_goal_scores  # noqa: F401  (re-export)
from engine.stats import pitman_morgan_test

__all__ = [
    "PassRateResult", "PASS_RATE_TARGETS", "compute_pass_rate",
    "TimeEfficiencyResult", "EFFICIENCY_BAND_RANGE", "efficiency_band", "compute_time_efficiency",
    "pass_rate_vs_time_frame",
    "compute_goal_scores", "compute_critical_alerts",  # re-exported from engine.scoring
    "DVH_UNIT_CATEGORIES", "compute_dvh_frame", "compute_dvh_consistency",
]

# Targets from docs/analysis_manual_th_v2.md, Module 1's table — Manual is
# the reference group and has no target of its own.
PASS_RATE_TARGETS: dict[PlanType, float] = {
    PlanType.AUTO: 50.0,
    PlanType.AUTO_MANUAL: 70.0,
}

_PLAN_ORDER = [p.value for p in PlanType]


@dataclass
class PassRateResult:
    per_patient: pd.DataFrame
    """One row per (pt_no, plan_type): n_total (the actual denominator),
    n_passed, pass_rate, plus the informational exclusion counts
    n_no_priority / n_non_evaluable / n_excluded_unmatched, and target /
    target_met (None for Manual, which has no target)."""

    cohort: pd.DataFrame
    """One row per plan_type: n (patients with a defined pass_rate), mean,
    sd, median, min, max, cv (%), and target_met_pct (of n_patients, the
    patients who have this plan at all)."""

    priority1: pd.DataFrame
    """Same shape as per_patient, restricted to Priority == 1 goals —
    computed with the same evaluable/matched filters (priority filtering
    is moot: every row here already has priority == 1)."""


def _patient_matched_goal_keys(patient_df: pd.DataFrame) -> set:
    """goal_keys present in every distinct plan_type this patient has."""
    plan_types = patient_df["plan_type"].unique()
    if len(plan_types) == 0:
        return set()
    key_sets = [set(patient_df.loc[patient_df["plan_type"] == pt, "goal_key"]) for pt in plan_types]
    return set.intersection(*key_sets)


def _filtered_rate(goals: pd.DataFrame) -> tuple[int, int, float]:
    n_total = len(goals)
    n_passed = int((goals["status"] == GoalStatus.PASS.value).sum())
    pass_rate = (n_passed / n_total * 100.0) if n_total > 0 else np.nan
    return n_total, n_passed, pass_rate


def compute_pass_rate(
    df: pd.DataFrame,
    *,
    exclude_non_evaluable: bool = True,
    exclude_no_priority: bool = True,
    matched_goals_only: bool = False,
) -> PassRateResult:
    """df is the (corrected, straight-pass-imputed) analysis frame — one
    row per goal per plan per patient, as engine.corrections.apply_corrections
    /engine.imputation.apply_straight_pass produce it."""
    empty_cols = ["pt_no", "plan_type", "n_total", "n_passed", "pass_rate",
                 "n_no_priority", "n_non_evaluable", "n_excluded_unmatched",
                 "target", "target_met"]
    if df.empty:
        empty = pd.DataFrame(columns=empty_cols)
        return PassRateResult(per_patient=empty, cohort=_cohort_summary(empty), priority1=empty.copy())

    rows, priority1_rows = [], []

    for pt_no, patient_df in df.groupby("pt_no", sort=False):
        matched_keys = _patient_matched_goal_keys(patient_df) if matched_goals_only else None

        for plan_type, plan_df in patient_df.groupby("plan_type", sort=False):
            remaining = plan_df

            n_no_priority = int(remaining["priority"].isna().sum())
            if exclude_no_priority:
                remaining = remaining[remaining["priority"].notna()]

            n_non_evaluable = int((~remaining["evaluable"]).sum())
            if exclude_non_evaluable:
                remaining = remaining[remaining["evaluable"]]

            if matched_goals_only:
                is_matched = remaining["goal_key"].isin(matched_keys)
                n_excluded_unmatched = int((~is_matched).sum())
                remaining = remaining[is_matched]
            else:
                n_excluded_unmatched = 0

            n_total, n_passed, pass_rate = _filtered_rate(remaining)
            target = PASS_RATE_TARGETS.get(PlanType(plan_type))
            target_met = (pass_rate >= target) if (target is not None and pd.notna(pass_rate)) else None

            rows.append(dict(
                pt_no=pt_no, plan_type=plan_type, n_total=n_total, n_passed=n_passed,
                pass_rate=pass_rate, n_no_priority=n_no_priority, n_non_evaluable=n_non_evaluable,
                n_excluded_unmatched=n_excluded_unmatched, target=target, target_met=target_met,
            ))

            # Priority-1 breakdown: same evaluable/matched filters, but the
            # priority filter is replaced by "priority == 1" outright.
            p1 = plan_df[plan_df["priority"] == 1]
            if exclude_non_evaluable:
                p1 = p1[p1["evaluable"]]
            if matched_goals_only:
                p1 = p1[p1["goal_key"].isin(matched_keys)]
            p1_total, p1_passed, p1_rate = _filtered_rate(p1)
            priority1_rows.append(dict(pt_no=pt_no, plan_type=plan_type,
                                       n_total=p1_total, n_passed=p1_passed, pass_rate=p1_rate))

    per_patient = pd.DataFrame(rows)
    priority1 = pd.DataFrame(priority1_rows)
    return PassRateResult(per_patient=per_patient, cohort=_cohort_summary(per_patient), priority1=priority1)


def _cohort_summary(per_patient: pd.DataFrame) -> pd.DataFrame:
    columns = ["plan_type", "n", "mean", "sd", "median", "min", "max", "cv",
              "target", "n_patients", "n_target_met", "target_met_pct"]
    if per_patient.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for plan_type in _PLAN_ORDER:
        g = per_patient[per_patient["plan_type"] == plan_type]
        if g.empty:
            continue
        rates = g["pass_rate"].dropna()
        n = len(rates)
        mean = rates.mean() if n else np.nan
        sd = rates.std(ddof=1) if n else np.nan  # NaN for n<2, matching pandas — an honest
                                                  # "not enough data", not a fabricated 0
        median = rates.median() if n else np.nan
        vmin = rates.min() if n else np.nan
        vmax = rates.max() if n else np.nan
        cv = (sd / abs(mean) * 100.0) if (n and pd.notna(mean) and mean != 0 and pd.notna(sd)) else np.nan

        target = PASS_RATE_TARGETS.get(PlanType(plan_type))
        n_target_met = int((g["target_met"] == True).sum()) if target is not None else None  # noqa: E712
        target_met_pct = (n_target_met / len(g) * 100.0) if (target is not None and len(g)) else None

        rows.append(dict(plan_type=plan_type, n=n, mean=mean, sd=sd, median=median, min=vmin, max=vmax,
                         cv=cv, target=target, n_patients=len(g), n_target_met=n_target_met,
                         target_met_pct=target_met_pct))
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# compute_time_efficiency
# --------------------------------------------------------------------------- #

# docs/analysis_manual_th_v2.md, Module 2's Efficiency Band table.
EFFICIENCY_BAND_RANGE = (40.0, 75.0)


@dataclass
class TimeEfficiencyResult:
    per_patient: pd.DataFrame
    """One row per pt_no: time_manual/time_auto/time_auto_manual (minutes),
    pct_eff_auto/pct_eff_auto_manual (None if either time involved is
    missing), and band_auto/band_auto_manual ("Below"/"Target"/"Exceeds",
    None alongside a None efficiency)."""

    cohort: pd.DataFrame
    """One row per plan_type (Auto, Auto+Manual): n (patients with a
    defined %efficiency), mean, sd, and n_below/n_target/n_exceeds."""


def efficiency_band(pct_eff: Optional[float]) -> Optional[str]:
    """"Below" / "Target" / "Exceeds" per the manual's table (40-75%
    inclusive is Target); None if pct_eff itself is None/NaN.

    Note for callers reading a band back out of a DataFrame column (as
    TimeEfficiencyResult.per_patient's band_auto/band_auto_manual are):
    once this None sits in a pandas column alongside other rows' band
    strings, pandas' string dtype coerces it to NaN — check with
    pd.isna(), not `is None`.
    """
    if pct_eff is None or pd.isna(pct_eff):
        return None
    low, high = EFFICIENCY_BAND_RANGE
    if pct_eff < low:
        return "Below"
    if pct_eff <= high:
        return "Target"
    return "Exceeds"


def _pct_efficiency(time_manual: Optional[float], time_plan: Optional[float]) -> Optional[float]:
    if pd.isna(time_manual) or pd.isna(time_plan) or time_manual == 0:
        return None
    return (time_manual - time_plan) / time_manual * 100.0


def compute_time_efficiency(df: pd.DataFrame) -> TimeEfficiencyResult:
    """df is the (corrected, straight-pass-imputed) analysis frame — same
    input as compute_pass_rate. planning_time_min is a per-plan value (one
    per patient×plan_type, repeated across that plan's goal rows), so this
    only needs the distinct combinations, not every goal row."""
    empty_per_patient = pd.DataFrame(columns=[
        "pt_no", "time_manual", "time_auto", "time_auto_manual",
        "pct_eff_auto", "pct_eff_auto_manual", "band_auto", "band_auto_manual",
    ])
    if df.empty:
        return TimeEfficiencyResult(per_patient=empty_per_patient,
                                    cohort=_time_efficiency_cohort(empty_per_patient))

    times = df[["pt_no", "plan_type", "planning_time_min"]].drop_duplicates()
    wide = times.pivot(index="pt_no", columns="plan_type", values="planning_time_min")
    wide = wide.reindex(columns=_PLAN_ORDER)

    rows = []
    for pt_no, row in wide.iterrows():
        t_manual = row.get(PlanType.MANUAL.value)
        t_auto = row.get(PlanType.AUTO.value)
        t_am = row.get(PlanType.AUTO_MANUAL.value)

        pct_auto = _pct_efficiency(t_manual, t_auto)
        pct_am = _pct_efficiency(t_manual, t_am)

        rows.append(dict(
            pt_no=pt_no, time_manual=t_manual, time_auto=t_auto, time_auto_manual=t_am,
            pct_eff_auto=pct_auto, pct_eff_auto_manual=pct_am,
            band_auto=efficiency_band(pct_auto), band_auto_manual=efficiency_band(pct_am),
        ))

    per_patient = pd.DataFrame(rows)
    return TimeEfficiencyResult(per_patient=per_patient, cohort=_time_efficiency_cohort(per_patient))


def _time_efficiency_cohort(per_patient: pd.DataFrame) -> pd.DataFrame:
    columns = ["plan_type", "n", "mean", "sd", "n_below", "n_target", "n_exceeds"]
    if per_patient.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for plan_type, pct_col, band_col in [
        (PlanType.AUTO.value, "pct_eff_auto", "band_auto"),
        (PlanType.AUTO_MANUAL.value, "pct_eff_auto_manual", "band_auto_manual"),
    ]:
        rates = per_patient[pct_col].dropna()
        n = len(rates)
        mean = rates.mean() if n else np.nan
        sd = rates.std(ddof=1) if n else np.nan
        n_below = int((per_patient[band_col] == "Below").sum())
        n_target = int((per_patient[band_col] == "Target").sum())
        n_exceeds = int((per_patient[band_col] == "Exceeds").sum())
        rows.append(dict(plan_type=plan_type, n=n, mean=mean, sd=sd,
                         n_below=n_below, n_target=n_target, n_exceeds=n_exceeds))
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# %Pass Rate vs Planning Time — the data behind Module 3's scatter plot
# --------------------------------------------------------------------------- #


def pass_rate_vs_time_frame(pass_rate_result: PassRateResult, df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pt_no, plan_type) with pass_rate and planning_time_min
    — every plan type, Manual included, per the manual's "colored/shaped
    by plan type" scatter spec. `df` is the same (corrected, imputed)
    analysis frame passed to compute_pass_rate/compute_time_efficiency.
    Rows missing either value are dropped (nothing to plot for them).
    """
    times = df[["pt_no", "plan_type", "planning_time_min"]].drop_duplicates()
    merged = pass_rate_result.per_patient[["pt_no", "plan_type", "pass_rate"]].merge(
        times, on=["pt_no", "plan_type"], how="left"
    )
    return merged.dropna(subset=["pass_rate", "planning_time_min"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# compute_dvh_frame — Module 3's DVH Consistency boxplot (§3.7)
# --------------------------------------------------------------------------- #

# GoalType -> the unit its AchievedValue is naturally in and the display
# category it belongs to. A dose is stored in cGy but displayed in Gy (the
# manual's own unit choice); a VolumeAtDose fraction (0-1) is displayed as a
# percent; AbsoluteVolumeAtDose is already cc, no conversion needed. Kept
# local (not reused from engine.corrections.GOAL_TYPE_UNITS) because that
# table's "cGy"/"%" labels are storage units for correction-entry, not the
# Gy-scaled display categories the manual's boxplot selector names.
DVH_UNIT_CATEGORIES: dict[str, tuple[str, float]] = {
    "DoseAtVolume": ("Dose (Gy)", 1.0 / 100.0),
    "DoseAtAbsoluteVolume": ("Dose (Gy)", 1.0 / 100.0),
    "AverageDose": ("Dose (Gy)", 1.0 / 100.0),
    "AbsoluteVolumeAtDose": ("Volume (cc)", 1.0),
    "VolumeAtDose": ("%Volume", 100.0),
}


def compute_dvh_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (patient, plan, goal) with achieved_value converted to
    its display unit. df must already be the corrected, straight-pass-
    imputed analysis frame (engine.corrections.apply_corrections then
    engine.imputation.apply_straight_pass — same precondition as every
    other compute_* function in this module) so a straight-pass patient's
    imputed Auto+Manual rows are included in the aggregate just like a
    real upload would be.

    Columns: patient, plan, goal_key, roi, structure_class, goal_type,
    goal_text, unit_category ("Dose (Gy)" | "%Volume" | "Volume (cc)"),
    value (in unit_category's unit), normalised_pct (achieved / goal *
    100 — always computable except for a zero-limit goal, where Goal = 0
    makes it undefined, or a missing achieved value).

    A row with no AchievedValue is dropped — there's nothing to plot for
    it; unlike compute_pass_rate/compute_goal_scores this doesn't filter
    on priority or "scorable" at all, since it's describing the raw DVH
    distribution, not a scored quality metric.
    """
    columns = ["patient", "plan", "goal_key", "roi", "structure_class", "goal_type",
              "goal_text", "unit_category", "value", "normalised_pct"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    have_value = df[df["achieved_value"].notna()].copy()
    if have_value.empty:
        return pd.DataFrame(columns=columns)

    unit_info = have_value["goal_type"].map(DVH_UNIT_CATEGORIES)
    have_value = have_value[unit_info.notna()].copy()
    if have_value.empty:
        return pd.DataFrame(columns=columns)
    unit_info = have_value["goal_type"].map(DVH_UNIT_CATEGORIES)
    have_value["unit_category"] = unit_info.map(lambda t: t[0])
    have_value["value"] = have_value["achieved_value"] * unit_info.map(lambda t: t[1])

    normalisable = have_value["acceptance_level"] != 0
    have_value["normalised_pct"] = np.where(
        normalisable, have_value["achieved_value"] / have_value["acceptance_level"] * 100.0, np.nan
    )

    out = have_value.rename(columns={"pt_no": "patient", "plan_type": "plan"})[columns]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# compute_dvh_consistency — §3.7's per-metric consistency table
# --------------------------------------------------------------------------- #

_DVH_PLAN_SLUG = {PlanType.MANUAL.value: "manual", PlanType.AUTO.value: "auto",
                  PlanType.AUTO_MANUAL.value: "auto_manual"}
_DVH_COMPARISON_PLANS = [PlanType.AUTO.value, PlanType.AUTO_MANUAL.value]


def _plan_descriptives(values: pd.Series) -> dict:
    n = int(values.notna().sum())
    vals = values.dropna()
    mean = vals.mean() if n else np.nan
    sd = vals.std(ddof=1) if n >= 2 else np.nan
    cv = (sd / abs(mean) * 100.0) if (n >= 2 and pd.notna(mean) and mean != 0 and pd.notna(sd)) else np.nan
    return dict(n=n, mean=mean, sd=sd, cv=cv, insufficient_n=n < 3)


def compute_dvh_consistency(dvh_frame: pd.DataFrame, *, value_col: str = "value") -> pd.DataFrame:
    """One row per goal_key: n/mean/sd/cv per plan (insufficient_n=True,
    sd=NaN when that plan has fewer than 3 patients for this goal — an
    honest "not enough data," never a misleading number), plus for Auto
    and Auto+Manual each vs Manual: sd_lower_than_manual, and the paired
    engine.stats.pitman_morgan_test / Wilcoxon signed-rank p-value (both
    None below 3 paired patients, or when the paired test is otherwise
    undefined — see pitman_morgan_test).

    `value_col` selects which of compute_dvh_frame()'s value columns to
    summarize — "value" (the goal's own display unit) or "normalised_pct".
    Pass compute_dvh_frame() output already filtered to one unit_category
    (or leave unfiltered when value_col="normalised_pct", since that
    column is unit-agnostic by construction).
    """
    base_columns = ["goal_key", "roi", "goal_text"]
    plan_columns = [f"{stat}_{slug}" for slug in _DVH_PLAN_SLUG.values()
                    for stat in ("n", "mean", "sd", "cv", "insufficient_n")]
    comparison_columns = [f"{stat}_{_DVH_PLAN_SLUG[p]}" for p in _DVH_COMPARISON_PLANS
                          for stat in ("n_pairs", "sd_lower_than_manual", "pm_t", "pm_p", "wilcoxon_p")]
    columns = base_columns + plan_columns + comparison_columns
    if dvh_frame.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for goal_key, g in dvh_frame.groupby("goal_key", sort=False):
        row = dict(goal_key=goal_key, roi=g["roi"].iloc[0], goal_text=g["goal_text"].iloc[0])

        plan_stats = {}
        for plan_type, slug in _DVH_PLAN_SLUG.items():
            stats_ = _plan_descriptives(g.loc[g["plan"] == plan_type, value_col])
            plan_stats[plan_type] = stats_
            for stat_name, stat_value in stats_.items():
                row[f"{stat_name}_{slug}"] = stat_value

        manual_vals = g.loc[g["plan"] == PlanType.MANUAL.value, ["patient", value_col]].dropna()
        for plan_type in _DVH_COMPARISON_PLANS:
            slug = _DVH_PLAN_SLUG[plan_type]
            plan_vals = g.loc[g["plan"] == plan_type, ["patient", value_col]].dropna()
            paired = manual_vals.merge(plan_vals, on="patient", suffixes=("_manual", "_plan"))
            n_pairs = len(paired)

            sd_manual, sd_plan = plan_stats[PlanType.MANUAL.value]["sd"], plan_stats[plan_type]["sd"]
            sd_lower = (sd_plan < sd_manual) if (pd.notna(sd_manual) and pd.notna(sd_plan)) else None

            pm = wilcoxon_p = None
            if n_pairs >= 3:
                pm = pitman_morgan_test(paired[f"{value_col}_plan"], paired[f"{value_col}_manual"])
                diffs = paired[f"{value_col}_plan"] - paired[f"{value_col}_manual"]
                if (diffs != 0).any():
                    wilcoxon_p = float(sps.wilcoxon(paired[f"{value_col}_plan"],
                                                    paired[f"{value_col}_manual"],
                                                    zero_method="zsplit").pvalue)

            row[f"n_pairs_{slug}"] = n_pairs
            row[f"sd_lower_than_manual_{slug}"] = sd_lower
            row[f"pm_t_{slug}"] = pm.t if pm else np.nan
            row[f"pm_p_{slug}"] = pm.p_value if pm else np.nan
            row[f"wilcoxon_p_{slug}"] = wilcoxon_p

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
