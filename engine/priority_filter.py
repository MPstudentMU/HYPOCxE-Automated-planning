"""
Module 4 – Priority Filtering
=============================
Filtering changes WHICH scored goals are summed. It never changes HOW a goal
is scored: base scores, multipliers and max scores still come from
engine/criteria_v0.py (frozen).

Quality Index (filtered) =
    Σ weighted_score (goals in selected priorities)
    / Σ max_weighted (goals in selected priorities) × 100

Expected input: the goal-level scoring table produced by engine/scoring.py,
one row per goal per plan per patient, with at least these columns:

    patient         str   e.g. "Pt1"
    plan            str   "Manual" | "Auto" | "Auto+Manual"
    priority        int   1 | 2 | 3   (NaN allowed for unscorable rows)
    category        str   "OAR Sparing" | "Target Coverage" | "Hot-spot Limit"
    base_score      float
    weighted_score  float  base_score × multiplier
    max_weighted    float  attainable max base × multiplier (V0 ceilings)
    scorable        bool   False for pending corrections / not evaluable
    structure_group str    optional, used by the radar chart
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

PRIORITY_LEVELS: tuple[int, ...] = (1, 2, 3)
PLAN_ORDER: list[str] = ["Manual", "Auto", "Auto+Manual"]
REQUIRED_COLUMNS = {
    "patient", "plan", "priority", "category",
    "base_score", "weighted_score", "max_weighted", "scorable",
}


# --------------------------------------------------------------------------- #
# Selection object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrioritySelection:
    """Immutable, validated set of priority levels to include."""

    levels: tuple[int, ...] = PRIORITY_LEVELS

    def __post_init__(self) -> None:
        cleaned = tuple(sorted({int(p) for p in self.levels}))
        if not cleaned:
            raise ValueError("Select at least one priority level.")
        invalid = set(cleaned) - set(PRIORITY_LEVELS)
        if invalid:
            raise ValueError(f"Invalid priority level(s): {sorted(invalid)}")
        object.__setattr__(self, "levels", cleaned)

    @classmethod
    def all(cls) -> "PrioritySelection":
        return cls(PRIORITY_LEVELS)

    @property
    def is_all(self) -> bool:
        return self.levels == PRIORITY_LEVELS

    @property
    def label(self) -> str:
        if self.is_all:
            return "All Priorities"
        return " + ".join(f"Priority {p}" for p in self.levels)

    @property
    def latex_set(self) -> str:
        """LaTeX description of the goal set, for the equation card."""
        if self.is_all:
            return r"i \in \text{all scorable goals}"
        levels = ", ".join(str(p) for p in self.levels)
        return rf"i \in \{{\text{{scorable goals with Priority }} {levels}\}}"


# --------------------------------------------------------------------------- #
# Core calculations
# --------------------------------------------------------------------------- #
def _validate(goal_scores: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(goal_scores.columns)
    if missing:
        raise KeyError(f"goal_scores is missing columns: {sorted(missing)}")


def _order_plans(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["plan"] = pd.Categorical(out["plan"], categories=PLAN_ORDER, ordered=True)
    return out


def filter_goal_scores(goal_scores: pd.DataFrame,
                       selection: PrioritySelection) -> pd.DataFrame:
    """Scorable goals whose priority is in the selection."""
    _validate(goal_scores)
    scorable = goal_scores[goal_scores["scorable"].astype(bool)]
    return scorable[scorable["priority"].isin(selection.levels)].copy()


def quality_index_table(goal_scores: pd.DataFrame,
                        selection: PrioritySelection) -> pd.DataFrame:
    """
    One row per patient × plan.
    Plans with no scorable goals in the selected priorities get
    quality_index = NaN (shown as "—"), never 0: an empty subset is not a
    bad score.
    """
    _validate(goal_scores)
    grid = goal_scores[["patient", "plan"]].drop_duplicates()
    filtered = filter_goal_scores(goal_scores, selection)

    agg = (filtered.groupby(["patient", "plan"], as_index=False)
                   .agg(n_goals=("weighted_score", "size"),
                        score=("weighted_score", "sum"),
                        max_score=("max_weighted", "sum")))

    out = grid.merge(agg, on=["patient", "plan"], how="left")
    out["n_goals"] = out["n_goals"].fillna(0).astype(int)
    denom = out["max_score"].where(out["max_score"] > 0)
    out["quality_index"] = out["score"] / denom * 100
    out["priority_filter"] = selection.label
    out = _order_plans(out)
    return out.sort_values(["patient", "plan"]).reset_index(drop=True)


def cohort_summary(qi_table: pd.DataFrame) -> pd.DataFrame:
    """n, mean, SD, median, min, max, CV of the (filtered) Quality Index."""
    rows = []
    for plan in PLAN_ORDER:
        s = qi_table.loc[qi_table["plan"] == plan, "quality_index"].dropna()
        mean = s.mean() if len(s) else np.nan
        sd = s.std(ddof=1) if len(s) > 1 else np.nan
        rows.append({
            "plan": plan, "n": len(s), "mean": mean, "sd": sd,
            "median": s.median() if len(s) else np.nan,
            "min": s.min() if len(s) else np.nan,
            "max": s.max() if len(s) else np.nan,
            "cv_pct": sd / abs(mean) * 100 if len(s) > 1 and mean else np.nan,
        })
    return pd.DataFrame(rows)


def category_breakdown(goal_scores: pd.DataFrame,
                       selection: PrioritySelection) -> pd.DataFrame:
    """Weighted score per patient × plan × category (filtered)."""
    filtered = filter_goal_scores(goal_scores, selection)
    out = (filtered.groupby(["patient", "plan", "category"], as_index=False)
                   ["weighted_score"].sum())
    return _order_plans(out).sort_values(["patient", "plan", "category"])


def radar_table(goal_scores: pd.DataFrame,
                selection: PrioritySelection) -> pd.DataFrame:
    """Mean base score per structure group × plan (filtered)."""
    if "structure_group" not in goal_scores.columns:
        return pd.DataFrame(columns=["structure_group", "plan", "mean_base"])
    filtered = filter_goal_scores(goal_scores, selection)
    filtered = filtered.dropna(subset=["structure_group"])
    out = (filtered.groupby(["structure_group", "plan"], as_index=False)
                   .agg(mean_base=("base_score", "mean")))
    return _order_plans(out)


def paired_consistency(qi_table: pd.DataFrame,
                       reference: str = "Manual") -> pd.DataFrame:
    """
    Pitman–Morgan (paired variance) and Wilcoxon signed-rank tests of each
    automated plan vs the reference, on the filtered Quality Index.
    Only patients with a value in BOTH plans are paired.
    """
    tbl = qi_table.assign(plan=qi_table["plan"].astype(str))
    wide = tbl.pivot_table(index="patient", columns="plan",
                           values="quality_index", aggfunc="first")
    rows = []
    for plan in [p for p in PLAN_ORDER if p != reference]:
        row = {"comparison": f"{plan} vs {reference}", "n_pairs": 0,
               "var_ratio": np.nan, "pm_t": np.nan, "pm_p": np.nan,
               "wilcoxon_p": np.nan}
        if plan in wide and reference in wide:
            pair = wide[[plan, reference]].dropna()
            x, y = pair[plan].to_numpy(float), pair[reference].to_numpy(float)
            n = len(pair)
            row["n_pairs"] = n
            if n >= 3:
                vy = np.var(y, ddof=1)
                row["var_ratio"] = np.var(x, ddof=1) / vy if vy > 0 else np.nan
                s, d = x + y, x - y
                if np.std(s) > 0 and np.std(d) > 0:
                    r = np.corrcoef(s, d)[0, 1]
                    if abs(r) < 1:
                        t = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
                        row["pm_t"] = t
                        row["pm_p"] = 2 * sps.t.sf(abs(t), df=n - 2)
                if np.any(d != 0):
                    row["wilcoxon_p"] = sps.wilcoxon(
                        x, y, zero_method="zsplit").pvalue
        rows.append(row)
    return pd.DataFrame(rows)
