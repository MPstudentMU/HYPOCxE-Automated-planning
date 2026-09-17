"""Tests for engine/analysis.py::compute_pass_rate(), against real pilot
fixtures (Pt1, Pt5 straight-pass) and small hand-built frames for the
filter/exclusion-count edge cases."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from engine import analysis as A
from engine import corrections as C
from engine import imputation as I
from engine import parser as P
from engine.schemas import DoseRegimen, FormInput, PlanType
from engine.storage import init_db, load_analysis_frame, save_case

FIXTURES = "tests/fixtures/pilot"


def _form(**overrides) -> FormInput:
    defaults = dict(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                    fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    defaults.update(overrides)
    return FormInput(**defaults)


def _pipeline(engine) -> pd.DataFrame:
    return I.apply_straight_pass(C.apply_corrections(load_analysis_frame(engine), engine))


@pytest.fixture
def pt1_pt5_engine():
    engine = init_db("sqlite://")
    parsed1 = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form(pt_no="Pt1", hn="90000001"))
    save_case(engine, _form(pt_no="Pt1", hn="90000001"), parsed1.plan_frames)

    parsed5 = P.parse_case_files([f"{FIXTURES}/5clinical_goals_90000005_auto.xlsx"],
                                 _form(pt_no="Pt5", hn="90000005"))
    save_case(engine, _form(pt_no="Pt5", hn="90000005"), parsed5.plan_frames)
    return engine


# --------------------------------------------------------------------------- #
# Real fixtures: Pt1 and Pt5 (straight-pass)
# --------------------------------------------------------------------------- #


def test_pt1_manual_pass_rate_hand_checked(pt1_pt5_engine):
    """Pt1 Manual (default filters): 6 raw goals -> Rectum_new has no
    priority (excluded), zBone/Bone Marrow has no AchievedValue (excluded)
    -> 4 considered, all PASS (BODY, Bladder, Kidney Lt, PTV) -> 100%."""
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine))
    row = result.per_patient[(result.per_patient.pt_no == "Pt1") &
                             (result.per_patient.plan_type == "Manual")].iloc[0]
    assert row.n_no_priority == 1
    assert row.n_non_evaluable == 1
    assert row.n_total == 4
    assert row.n_passed == 4
    assert row.pass_rate == pytest.approx(100.0)
    assert pd.isna(row.target) and row.target_met is None  # Manual has no target


def test_pt1_auto_pass_rate_and_target(pt1_pt5_engine):
    """Pt1 Auto: 3 goals (BODY, Bladder, ITV), all PASS, none excluded."""
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine))
    row = result.per_patient[(result.per_patient.pt_no == "Pt1") &
                             (result.per_patient.plan_type == "Auto")].iloc[0]
    assert row.n_total == 3
    assert row.n_passed == 3
    assert row.pass_rate == pytest.approx(100.0)
    assert row.target == 50.0
    assert row.target_met is True


def test_pt1_gets_straight_pass_auto_manual_row(pt1_pt5_engine):
    """Pt1 uploaded no Auto+Manual file -> straight-pass copies Auto ->
    Auto+Manual's pass rate must equal Auto's."""
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine))
    per_patient = result.per_patient[result.per_patient.pt_no == "Pt1"]
    auto = per_patient[per_patient.plan_type == "Auto"].iloc[0]
    am = per_patient[per_patient.plan_type == "Auto+Manual"].iloc[0]
    assert am.pass_rate == pytest.approx(auto.pass_rate)
    assert am.target == 70.0


def test_pt5_straight_pass_rate_hand_checked(pt1_pt5_engine):
    """Pt5 Auto (from the real fixture): 5 unique rows after dedup, one
    (Bone) has no AchievedValue -> 4 considered; of those, BODY/Bladder
    PASS, Femur Head Lt/Cauda Equina... check against the fixture's own
    Status column: BODY PASS, Bladder PASS, Femur Head Lt FAIL, Cauda
    Equina PASS -> 3 of 4 PASS -> 75%."""
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine))
    row = result.per_patient[(result.per_patient.pt_no == "Pt5") &
                             (result.per_patient.plan_type == "Auto")].iloc[0]
    assert row.n_non_evaluable == 1
    assert row.n_total == 4
    assert row.n_passed == 3
    assert row.pass_rate == pytest.approx(75.0)
    assert row.target_met is True  # 75% >= 50%


def test_pt5_auto_manual_matches_auto_after_straight_pass(pt1_pt5_engine):
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine))
    per_patient = result.per_patient[result.per_patient.pt_no == "Pt5"]
    auto = per_patient[per_patient.plan_type == "Auto"].iloc[0]
    am = per_patient[per_patient.plan_type == "Auto+Manual"].iloc[0]
    assert am.pass_rate == pytest.approx(auto.pass_rate) == pytest.approx(75.0)
    assert am.target == 70.0
    assert am.target_met is True  # 75% >= 70%


def test_priority1_breakdown_hand_checked(pt1_pt5_engine):
    """Pt1 Manual priority-1 goals: BODY, Bladder, PTV (Kidney Lt is
    priority 2, Rectum has no priority, Bone Marrow is non-evaluable and
    also priority 1 but excluded by the evaluable filter) -> 3 goals, all
    PASS -> 100%."""
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine))
    row = result.priority1[(result.priority1.pt_no == "Pt1") &
                           (result.priority1.plan_type == "Manual")].iloc[0]
    assert row.n_total == 3
    assert row.n_passed == 3
    assert row.pass_rate == pytest.approx(100.0)


def test_cohort_summary_hand_checked(pt1_pt5_engine):
    """Auto pass rates: Pt1=100, Pt5=75 -> mean=87.5, sd=stdev([100,75])
    (ddof=1) = 17.6776695..., median=87.5, min=75, max=100,
    cv=sd/87.5*100."""
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine))
    row = result.cohort[result.cohort.plan_type == "Auto"].iloc[0]
    assert row.n == 2
    assert row["mean"] == pytest.approx(87.5)
    assert row["sd"] == pytest.approx(math.sqrt(2) * 12.5, abs=1e-6)  # stdev([100,75], ddof=1)
    assert row["median"] == pytest.approx(87.5)
    assert row["min"] == pytest.approx(75.0)
    assert row["max"] == pytest.approx(100.0)
    assert row["cv"] == pytest.approx(row["sd"] / 87.5 * 100.0)
    assert row.n_patients == 2
    assert row.n_target_met == 2  # both Pt1 (100%) and Pt5 (75%) clear 50%
    assert row.target_met_pct == pytest.approx(100.0)


def test_matched_goals_only_restricts_to_shared_goal_keys(pt1_pt5_engine):
    """Pt1's Manual/Auto/Auto+Manual(imputed) share only BODY and Bladder
    by goal_key — everything else gets excluded as unmatched."""
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine), matched_goals_only=True)
    manual = result.per_patient[(result.per_patient.pt_no == "Pt1") &
                                (result.per_patient.plan_type == "Manual")].iloc[0]
    auto = result.per_patient[(result.per_patient.pt_no == "Pt1") &
                              (result.per_patient.plan_type == "Auto")].iloc[0]
    assert manual.n_total == 2  # BODY, Bladder only (Kidney Lt, PTV excluded)
    assert manual.n_excluded_unmatched == 2
    assert auto.n_total == 2  # BODY, Bladder only (ITV excluded)
    assert auto.n_excluded_unmatched == 1


def test_matched_goals_only_default_is_off(pt1_pt5_engine):
    result = A.compute_pass_rate(_pipeline(pt1_pt5_engine))
    row = result.per_patient[(result.per_patient.pt_no == "Pt1") &
                             (result.per_patient.plan_type == "Manual")].iloc[0]
    assert row.n_excluded_unmatched == 0


# --------------------------------------------------------------------------- #
# Toggle behavior on a small, hand-built frame
# --------------------------------------------------------------------------- #


def _row(pt_no, plan_type, goal_key, priority, status, evaluable=True):
    return dict(pt_no=pt_no, plan_type=plan_type, goal_key=goal_key, priority=priority,
               status=status, evaluable=evaluable)


def test_exclude_non_evaluable_toggle_off_includes_it_in_denominator():
    df = pd.DataFrame([
        _row("Pt1", "Manual", "g1", 1, "PASS"),
        _row("Pt1", "Manual", "g2", 1, None, evaluable=False),
    ])
    with_exclusion = A.compute_pass_rate(df, exclude_non_evaluable=True)
    without_exclusion = A.compute_pass_rate(df, exclude_non_evaluable=False)

    r1 = with_exclusion.per_patient.iloc[0]
    assert r1.n_total == 1 and r1.pass_rate == pytest.approx(100.0) and r1.n_non_evaluable == 1

    r2 = without_exclusion.per_patient.iloc[0]
    assert r2.n_total == 2  # non-evaluable row counted in the denominator
    assert r2.n_passed == 1  # but it isn't PASS, so numerator unaffected
    assert r2.pass_rate == pytest.approx(50.0)
    assert r2.n_non_evaluable == 1  # still reported, even though not excluded


def test_exclude_no_priority_toggle_off_includes_it_in_denominator():
    df = pd.DataFrame([
        _row("Pt1", "Manual", "g1", 1, "PASS"),
        _row("Pt1", "Manual", "g2", None, "FAIL"),
    ])
    without_exclusion = A.compute_pass_rate(df, exclude_no_priority=False)
    r = without_exclusion.per_patient.iloc[0]
    assert r.n_total == 2
    assert r.n_no_priority == 1
    assert r.pass_rate == pytest.approx(50.0)


def test_zero_denominator_is_nan_not_zero():
    """A plan with nothing left after filtering reports pass_rate as NaN
    (undefined), never a misleading 0%."""
    df = pd.DataFrame([_row("Pt1", "Manual", "g1", None, "PASS")])
    result = A.compute_pass_rate(df, exclude_no_priority=True)
    row = result.per_patient.iloc[0]
    assert row.n_total == 0
    assert math.isnan(row.pass_rate)
    assert row.target_met is None


def test_compute_pass_rate_empty_df():
    result = A.compute_pass_rate(pd.DataFrame(columns=["pt_no", "plan_type", "goal_key",
                                                        "priority", "status", "evaluable"]))
    assert result.per_patient.empty
    assert result.cohort.empty
    assert result.priority1.empty


# =========================================================================== #
# compute_time_efficiency
# =========================================================================== #


@pytest.fixture
def pt1_pt5_engine_with_times():
    """Same pilot files as pt1_pt5_engine, but with planning times set so
    %efficiency is actually computable: Pt1 Manual=100min, Auto=25min
    (-> exactly 75%, the Target/Exceeds boundary); Pt5 has no Manual plan
    at all, so no %efficiency is computable for it."""
    engine = init_db("sqlite://")
    parsed1 = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form(pt_no="Pt1", hn="90000001", time_manual=100.0, time_auto=25.0))
    save_case(engine, _form(pt_no="Pt1", hn="90000001", time_manual=100.0, time_auto=25.0),
             parsed1.plan_frames)

    parsed5 = P.parse_case_files([f"{FIXTURES}/5clinical_goals_90000005_auto.xlsx"],
                                 _form(pt_no="Pt5", hn="90000005", time_auto=30.0))
    save_case(engine, _form(pt_no="Pt5", hn="90000005", time_auto=30.0), parsed5.plan_frames)
    return engine


def test_pt1_time_efficiency_hand_checked(pt1_pt5_engine_with_times):
    """Pt1: Manual=100min, Auto=25min -> (100-25)/100*100 = 75.0% exactly
    -> Target (40-75% is inclusive of 75). Auto+Manual is straight-pass
    imputed from Auto (Pt1 uploaded no real Auto+Manual file) -> same
    25min -> same 75.0% -> same Target band."""
    result = A.compute_time_efficiency(_pipeline(pt1_pt5_engine_with_times))
    row = result.per_patient[result.per_patient.pt_no == "Pt1"].iloc[0]

    assert row.time_manual == 100.0
    assert row.time_auto == 25.0
    assert row.time_auto_manual == 25.0  # straight-pass: same as Auto
    assert row.pct_eff_auto == pytest.approx(75.0)
    assert row.pct_eff_auto_manual == pytest.approx(75.0)
    assert row.band_auto == "Target"
    assert row.band_auto_manual == "Target"


def test_pt5_time_efficiency_none_without_manual_time(pt1_pt5_engine_with_times):
    """Pt5 has no Manual plan at all -> time_manual is NaN -> %efficiency
    is None (undefined), not 0 or some fallback."""
    result = A.compute_time_efficiency(_pipeline(pt1_pt5_engine_with_times))
    row = result.per_patient[result.per_patient.pt_no == "Pt5"].iloc[0]

    assert pd.isna(row.time_manual)
    assert row.time_auto == 30.0
    assert row.time_auto_manual == 30.0  # straight-pass still copies the time
    assert pd.isna(row.pct_eff_auto)
    assert pd.isna(row.pct_eff_auto_manual)
    # pandas' string dtype coerces a None sitting alongside other rows'
    # "Target"/"Below"/"Exceeds" strings into NaN once in a DataFrame
    # column — check with pd.isna(), not `is None` (see efficiency_band's
    # docstring).
    assert pd.isna(row.band_auto)
    assert pd.isna(row.band_auto_manual)


def test_time_efficiency_cohort_hand_checked(pt1_pt5_engine_with_times):
    """Only Pt1 has a defined %efficiency (75.0) for Auto/Auto+Manual ->
    n=1, mean=75.0, sd undefined (n<2), 1 patient in the Target band."""
    result = A.compute_time_efficiency(_pipeline(pt1_pt5_engine_with_times))
    for plan_type in ("Auto", "Auto+Manual"):
        row = result.cohort[result.cohort.plan_type == plan_type].iloc[0]
        assert row.n == 1
        assert row["mean"] == pytest.approx(75.0)
        assert pd.isna(row["sd"])
        assert row.n_below == 0
        assert row.n_target == 1
        assert row.n_exceeds == 0


# --------------------------------------------------------------------------- #
# efficiency_band — hand-built values covering every band and the boundaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pct_eff,expected", [
    (None, None),
    (float("nan"), None),
    (0.0, "Below"),
    (39.9, "Below"),
    (40.0, "Target"),   # lower boundary is inclusive
    (57.5, "Target"),
    (75.0, "Target"),   # upper boundary is inclusive
    (75.1, "Exceeds"),
    (100.0, "Exceeds"),
])
def test_efficiency_band(pct_eff, expected):
    assert A.efficiency_band(pct_eff) == expected


# --------------------------------------------------------------------------- #
# compute_time_efficiency — hand-built frames for band/edge-case coverage
# --------------------------------------------------------------------------- #


def _time_row(pt_no, plan_type, planning_time_min):
    return dict(pt_no=pt_no, plan_type=plan_type, planning_time_min=planning_time_min)


def test_below_and_exceeds_bands_hand_checked():
    df = pd.DataFrame([
        _time_row("PtBelow", "Manual", 100.0), _time_row("PtBelow", "Auto", 70.0),   # 30% -> Below
        _time_row("PtBelow", "Auto+Manual", 70.0),
        _time_row("PtExceeds", "Manual", 100.0), _time_row("PtExceeds", "Auto", 10.0),  # 90% -> Exceeds
        _time_row("PtExceeds", "Auto+Manual", 10.0),
    ])
    result = A.compute_time_efficiency(df)
    below = result.per_patient[result.per_patient.pt_no == "PtBelow"].iloc[0]
    exceeds = result.per_patient[result.per_patient.pt_no == "PtExceeds"].iloc[0]

    assert below.pct_eff_auto == pytest.approx(30.0)
    assert below.band_auto == "Below"
    assert exceeds.pct_eff_auto == pytest.approx(90.0)
    assert exceeds.band_auto == "Exceeds"

    cohort_auto = result.cohort[result.cohort.plan_type == "Auto"].iloc[0]
    assert cohort_auto.n_below == 1
    assert cohort_auto.n_exceeds == 1
    assert cohort_auto.n_target == 0
    assert cohort_auto.n == 2
    assert cohort_auto["mean"] == pytest.approx((30.0 + 90.0) / 2)


def test_zero_manual_time_is_undefined_not_division_error():
    df = pd.DataFrame([_time_row("Pt1", "Manual", 0.0), _time_row("Pt1", "Auto", 10.0)])
    result = A.compute_time_efficiency(df)
    row = result.per_patient.iloc[0]
    assert pd.isna(row.pct_eff_auto)
    assert row.band_auto is None


def test_missing_auto_time_is_none_not_zero():
    df = pd.DataFrame([_time_row("Pt1", "Manual", 100.0)])  # no Auto row at all
    result = A.compute_time_efficiency(df)
    row = result.per_patient.iloc[0]
    assert pd.isna(row.time_auto)
    assert pd.isna(row.pct_eff_auto)
    assert row.band_auto is None


def test_compute_time_efficiency_empty_df():
    result = A.compute_time_efficiency(pd.DataFrame(columns=["pt_no", "plan_type", "planning_time_min"]))
    assert result.per_patient.empty
    assert result.cohort.empty


# =========================================================================== #
# pass_rate_vs_time_frame
# =========================================================================== #


def test_pass_rate_vs_time_frame_hand_checked(pt1_pt5_engine_with_times):
    df = _pipeline(pt1_pt5_engine_with_times)
    pr = A.compute_pass_rate(df)
    scatter = A.pass_rate_vs_time_frame(pr, df)

    pt1_auto = scatter[(scatter.pt_no == "Pt1") & (scatter.plan_type == "Auto")].iloc[0]
    assert pt1_auto.pass_rate == pytest.approx(100.0)
    assert pt1_auto.planning_time_min == pytest.approx(25.0)

    # Pt5 has no Manual plan at all -> no Manual point in the scatter data
    assert "Manual" not in set(scatter[scatter.pt_no == "Pt5"]["plan_type"])
    # every plan type is represented overall (Manual included, per the manual's spec)
    assert set(scatter["plan_type"]) == {"Manual", "Auto", "Auto+Manual"}


def test_pass_rate_vs_time_frame_drops_rows_missing_either_value():
    pr_per_patient = pd.DataFrame([dict(pt_no="Pt1", plan_type="Auto", pass_rate=80.0)])

    class _FakeResult:
        per_patient = pr_per_patient

    df = pd.DataFrame([dict(pt_no="Pt1", plan_type="Auto", planning_time_min=None)])
    scatter = A.pass_rate_vs_time_frame(_FakeResult(), df)
    assert scatter.empty
