"""Statistical tests and effect sizes.

mcnemar_test() / mcnemar_manual_vs_auto(): scipy has no built-in McNemar's
test, so this implements it directly on the two off-diagonal counts of a
2x2 paired table — the exact binomial test for a small number of
discordant pairs (matches statsmodels' own default convention), else the
continuity-corrected chi-square approximation.

correlation_pass_rate_vs_time(): Pearson's r and Spearman's ρ (each with
its p-value) plus a linear-fit line, for Module 3's %Pass Rate vs Planning
Time scatter (docs/analysis_manual_th_v2.md, Module 2's visualization
table). Per the manual's own caption: points from the same patient (one
per plan type) aren't fully independent, so this is for exploring a trend,
not for inference — callers should show that caveat next to the numbers,
not just the numbers.

pitman_morgan_test(): §3.7's paired-variance-equality test. This is the
same formula engine/priority_filter.py's paired_consistency() already has
inlined for the single aggregate Quality Index (that function is
untouched here — it's already shipped and tested); this reusable version
is for engine.analysis.compute_dvh_consistency(), which needs the same
test run once per DVH metric rather than once for Quality Index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine.schemas import GoalStatus, PlanType

__all__ = [
    "McNemarResult", "mcnemar_test", "mcnemar_manual_vs_auto",
    "CorrelationResult", "correlation_pass_rate_vs_time",
    "PitmanMorganResult", "pitman_morgan_test",
]


@dataclass
class McNemarResult:
    b: int
    """Discordant pairs where the first condition passed and the second failed."""
    c: int
    """Discordant pairs where the first condition failed and the second passed."""
    n_discordant: int
    n_pairs: int
    """Total paired goals considered, concordant pairs included."""
    statistic: Optional[float]
    """The chi-square statistic — only set for the asymptotic method."""
    p_value: Optional[float]
    method: str


def mcnemar_test(b: int, c: int, *, n_pairs: Optional[int] = None,
                 exact_threshold: int = 25) -> McNemarResult:
    """McNemar's test on a 2x2 table's off-diagonal counts. Uses the exact
    binomial test when there are fewer than `exact_threshold` discordant
    pairs (b + c) — the common convention, matching statsmodels' own
    default — else the continuity-corrected chi-square approximation.
    `n_pairs`, if given, is reported alongside as the total paired N
    (concordant pairs included); defaults to b + c.
    """
    n_discordant = b + c
    total_pairs = n_pairs if n_pairs is not None else n_discordant

    if n_discordant == 0:
        return McNemarResult(b=b, c=c, n_discordant=0, n_pairs=total_pairs,
                             statistic=None, p_value=1.0, method="no discordant pairs")

    if n_discordant < exact_threshold:
        result = sps.binomtest(min(b, c), n_discordant, p=0.5, alternative="two-sided")
        return McNemarResult(b=b, c=c, n_discordant=n_discordant, n_pairs=total_pairs,
                             statistic=None, p_value=float(result.pvalue), method="exact")

    statistic = (abs(b - c) - 1) ** 2 / n_discordant
    p_value = sps.chi2.sf(statistic, df=1)
    return McNemarResult(b=b, c=c, n_discordant=n_discordant, n_pairs=total_pairs,
                         statistic=float(statistic), p_value=float(p_value),
                         method="asymptotic (continuity-corrected)")


def mcnemar_manual_vs_auto(df: pd.DataFrame) -> Optional[McNemarResult]:
    """Pairs Manual and Auto goals by (pt_no, goal_key) — same clinical
    goal, same patient, both plans — and runs McNemar's test on their
    PASS/FAIL agreement. Only pairs where both sides have a resolved
    Status are used (a pending/non-evaluable goal can't be paired).
    None if there is nothing pairable at all.
    """
    manual = df.loc[df["plan_type"] == PlanType.MANUAL.value, ["pt_no", "goal_key", "status"]]
    auto = df.loc[df["plan_type"] == PlanType.AUTO.value, ["pt_no", "goal_key", "status"]]
    paired = manual.merge(auto, on=["pt_no", "goal_key"], suffixes=("_manual", "_auto"))
    paired = paired.dropna(subset=["status_manual", "status_auto"])
    if paired.empty:
        return None

    manual_pass = paired["status_manual"] == GoalStatus.PASS.value
    auto_pass = paired["status_auto"] == GoalStatus.PASS.value
    b = int((manual_pass & ~auto_pass).sum())
    c = int((~manual_pass & auto_pass).sum())
    return mcnemar_test(b, c, n_pairs=len(paired))


@dataclass
class CorrelationResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    slope: float
    """Linear fit: pass_rate = slope * planning_time + intercept."""
    intercept: float


def correlation_pass_rate_vs_time(planning_time: pd.Series, pass_rate: pd.Series) -> Optional[CorrelationResult]:
    """Pearson's r (with its linear-fit line, from the same regression) and
    Spearman's ρ between planning time and %Pass Rate, across whatever
    points are given — typically every (patient, plan type) point pooled
    together, per the manual's spec. None if there are fewer than 3 paired
    points, or planning_time is constant (a fit/correlation is undefined
    either way).
    """
    paired = pd.DataFrame({"x": planning_time, "y": pass_rate}).dropna()
    if len(paired) < 3 or paired["x"].nunique() < 2:
        return None

    fit = sps.linregress(paired["x"], paired["y"])
    spearman_rho, spearman_p = sps.spearmanr(paired["x"], paired["y"])
    return CorrelationResult(
        n=len(paired), pearson_r=float(fit.rvalue), pearson_p=float(fit.pvalue),
        spearman_rho=float(spearman_rho), spearman_p=float(spearman_p),
        slope=float(fit.slope), intercept=float(fit.intercept),
    )


@dataclass
class PitmanMorganResult:
    n: int
    r: float
    """Corr(X+Y, X-Y)."""
    t: float
    df: int
    p_value: float


def pitman_morgan_test(x: pd.Series, y: pd.Series) -> Optional[PitmanMorganResult]:
    """Pitman-Morgan test for equality of variances in paired data
    (docs/analysis_manual_th_v2.md §3.7): r = Corr(X+Y, X-Y),
    t = r * sqrt(n-2) / sqrt(1-r^2), df = n-2, p = 2 * P(T > |t|). Chosen
    over Levene's test because the data is paired (same patient, two
    plans), which violates Levene's independence assumption.

    None when there are fewer than 3 pairs (undefined), or r is undefined
    (X+Y or X-Y is constant) or ±1 (t is a division by zero) — each a
    real "can't test this," not a 0 or 1 masquerading as an answer.
    """
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(paired)
    if n < 3:
        return None

    s = paired["x"] + paired["y"]
    d = paired["x"] - paired["y"]
    if s.std() == 0 or d.std() == 0:
        return None
    r = float(np.corrcoef(s, d)[0, 1])
    if abs(r) >= 1:
        return None

    t = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
    p_value = 2 * sps.t.sf(abs(t), df=n - 2)
    return PitmanMorganResult(n=n, r=r, t=float(t), df=n - 2, p_value=float(p_value))
