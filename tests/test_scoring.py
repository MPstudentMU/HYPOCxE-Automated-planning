"""Tests for engine/scoring.py: score_goal, compute_goal_scores,
compute_critical_alerts — against docs/analysis_manual_th_v2.md §3.5's
worked example, hand-built boundary/edge cases, and real Pt1/Pt5 pilot
fixtures."""
from __future__ import annotations

import pandas as pd
import pytest

from engine import corrections as C
from engine import imputation as I
from engine import parser as P
from engine import scoring as S
from engine.priority_filter import REQUIRED_COLUMNS as PRIORITY_FILTER_REQUIRED_COLUMNS
from engine.schemas import DoseRegimen, FormInput
from engine.storage import init_db, load_analysis_frame, save_case

FIXTURES = "tests/fixtures/pilot"


def _row(**overrides) -> pd.Series:
    defaults = dict(
        roi="Bladder", criteria="AtMost", goal_type="DoseAtVolume",
        acceptance_level=100.0, achieved_value=100.0, priority=1,
        evaluable=True, structure_class="OAR",
    )
    defaults.update(overrides)
    return pd.Series(defaults)


def _form(**overrides) -> FormInput:
    defaults = dict(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                    fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    defaults.update(overrides)
    return FormInput(**defaults)


def _pipeline(engine) -> pd.DataFrame:
    return I.apply_straight_pass(C.apply_corrections(load_analysis_frame(engine), engine))


# =========================================================================== #
# §3.5's worked example — every row, hand-checked against the manual's own
# published Weighted Score
# =========================================================================== #


def test_worked_example_row1_ptv_coverage():
    """PTV V4275cGy >= 95%: Goal 95.00%, Achieved 99.87% -> +5.13%, base
    +3, P1 Target x5 -> weighted +15."""
    row = _row(roi="PTV", criteria="AtLeast", goal_type="VolumeAtDose",
              acceptance_level=0.9500, achieved_value=0.9987, priority=1, structure_class="TARGET")
    scored = S.score_goal(row)
    assert scored.category == "Target Coverage"
    assert scored.improvement_pct == pytest.approx(5.13, abs=0.01)
    assert scored.base_score == pytest.approx(3.0)
    assert scored.multiplier == pytest.approx(5.0)
    assert scored.weighted_score == pytest.approx(15.0)


def test_worked_example_row2_rectum_oar():
    """Rectum V4000cGy <= 75%: Goal 75%, Achieved 60% -> +20.00%, base +3,
    P1 OAR x3 -> weighted +9."""
    row = _row(roi="Rectum", criteria="AtMost", goal_type="VolumeAtDose",
              acceptance_level=0.75, achieved_value=0.60, priority=1, structure_class="OAR")
    scored = S.score_goal(row)
    assert scored.category == "OAR Sparing"
    assert scored.improvement_pct == pytest.approx(20.0, abs=0.01)
    assert scored.base_score == pytest.approx(3.0)
    assert scored.weighted_score == pytest.approx(9.0)


def test_worked_example_row3_body_hotspot_capped():
    """BODY D0.0% <= 4820cGy: Goal 4820, Achieved 4700 -> +2.49%, natural
    OAR base would be +1 already (2-5% band); Hot-spot caps at +1 either
    way, P1 x3 -> weighted +3."""
    row = _row(roi="BODY", criteria="AtMost", goal_type="DoseAtVolume",
              acceptance_level=4820.0, achieved_value=4700.0, priority=1, structure_class="OAR")
    scored = S.score_goal(row)
    assert scored.category == "Hot-spot Limit"
    assert scored.improvement_pct == pytest.approx(2.49, abs=0.01)
    assert scored.base_score == pytest.approx(1.0)
    assert scored.weighted_score == pytest.approx(3.0)


def test_worked_example_row4_bladder_unspecified_band():
    """Bladder D0.03cc <= 4725cGy: Goal 4725, Achieved 4900 -> -3.70%,
    lands in the "-5..-2" unspecified-by-protocol band, fixed at base 0
    -> weighted 0."""
    row = _row(roi="Bladder", criteria="AtMost", goal_type="DoseAtAbsoluteVolume",
              acceptance_level=4725.0, achieved_value=4900.0, priority=1, structure_class="OAR")
    scored = S.score_goal(row)
    assert scored.category == "OAR Sparing"
    assert scored.improvement_pct == pytest.approx(-3.70, abs=0.01)
    assert scored.base_score == pytest.approx(0.0)
    assert scored.weighted_score == pytest.approx(0.0)


def test_worked_example_row5_rectum_zero_limit_achieved_above_zero():
    """Rectum V4725cGy <= 0.0% (zero-limit): Achieved 0.09% (> 0) -> base
    0 (not +1) per the corrected reading — the Improvement % formula is
    not used at all (Goal = 0), so improvement_pct is None."""
    row = _row(roi="Rectum", criteria="AtMost", goal_type="VolumeAtDose",
              acceptance_level=0.0, achieved_value=0.0009, priority=1, structure_class="OAR")
    scored = S.score_goal(row)
    assert scored.category == "OAR Sparing"
    assert scored.is_zero_limit is True
    assert scored.improvement_pct is None
    assert scored.base_score == pytest.approx(0.0)
    assert scored.weighted_score == pytest.approx(0.0)
    assert scored.max_base == pytest.approx(1.0)  # not 3 — §3.4's v3.0 correction


# =========================================================================== #
# Zero-limit goals (§3.4) — boundary at exactly 0
# =========================================================================== #


def test_zero_limit_achieved_exactly_zero_gets_plus_one():
    row = _row(roi="Bladder", criteria="AtMost", goal_type="VolumeAtDose",
              acceptance_level=0.0, achieved_value=0.0, priority=2, structure_class="OAR")
    scored = S.score_goal(row)
    assert scored.is_zero_limit is True
    assert scored.base_score == pytest.approx(1.0)
    assert scored.max_base == pytest.approx(1.0)
    assert scored.weighted_score == pytest.approx(1.0 * 2.0)  # P2 OAR x2


def test_zero_limit_only_applies_to_atmost():
    """A hypothetical AtLeast goal with acceptance_level == 0 is outside
    the specified rules — refuses to guess rather than divide by zero."""
    row = _row(roi="PTV", criteria="AtLeast", goal_type="VolumeAtDose",
              acceptance_level=0.0, achieved_value=0.5, priority=1, structure_class="TARGET")
    with pytest.raises(ValueError):
        S.score_goal(row)


# =========================================================================== #
# Coverage-ceiling goals
# =========================================================================== #


def test_coverage_ceiling_max_base_is_zero():
    row = _row(roi="ITV", criteria="AtLeast", goal_type="VolumeAtDose",
              acceptance_level=1.0, achieved_value=1.0, priority=1, structure_class="TARGET")
    scored = S.score_goal(row)
    assert scored.is_coverage_ceiling is True
    assert scored.max_base == pytest.approx(0.0)
    # the score itself still runs the normal formula — not forced to 0
    assert scored.improvement_pct == pytest.approx(0.0)
    assert scored.base_score == pytest.approx(0.0)


def test_coverage_ceiling_gated_to_volume_at_dose_only():
    """A DoseAtVolume AtLeast goal's AcceptanceLevel is a dose (e.g. 4500
    cGy) — a raw ">= 1.0" check would misfire on any such goal. Confirms
    it doesn't."""
    row = _row(roi="PTV", criteria="AtLeast", goal_type="DoseAtVolume",
              acceptance_level=4500.0, achieved_value=4550.0, priority=1, structure_class="TARGET")
    scored = S.score_goal(row)
    assert scored.is_coverage_ceiling is False
    assert scored.max_base == pytest.approx(3.0)


def test_coverage_ceiling_below_100_is_not_a_ceiling_goal():
    row = _row(roi="ITV", criteria="AtLeast", goal_type="VolumeAtDose",
              acceptance_level=0.95, achieved_value=0.97, priority=1, structure_class="TARGET")
    scored = S.score_goal(row)
    assert scored.is_coverage_ceiling is False
    assert scored.max_base == pytest.approx(3.0)


# =========================================================================== #
# Classification (§3.1)
# =========================================================================== #


@pytest.mark.parametrize("criteria,roi,structure_class,expected_category", [
    ("AtLeast", "PTV", "TARGET", "Target Coverage"),
    ("AtLeast", "Bladder", "OAR", "Target Coverage"),  # AtLeast always -> Target, regardless of roi
    ("AtMost", "BODY", "OAR", "Hot-spot Limit"),
    ("AtMost", "PTV", "TARGET", "Hot-spot Limit"),
    ("AtMost", "PTV-N", "TARGET", "Hot-spot Limit"),
    ("AtMost", "Bladder", "OAR", "OAR Sparing"),
    ("AtMost", "Kidney Lt", "OAR", "OAR Sparing"),
])
def test_classification(criteria, roi, structure_class, expected_category):
    row = _row(criteria=criteria, roi=roi, structure_class=structure_class,
              goal_type="DoseAtVolume", acceptance_level=100.0, achieved_value=90.0)
    scored = S.score_goal(row)
    assert scored.category == expected_category


# =========================================================================== #
# Non-scorable (§3.4)
# =========================================================================== #


def test_missing_achieved_value_is_non_scorable():
    row = _row(evaluable=False, achieved_value=None)
    scored = S.score_goal(row)
    assert scored.scorable is False
    assert scored.base_score is None
    assert scored.weighted_score is None
    assert scored.max_weighted is None
    assert scored.category == "OAR Sparing"  # still classified — a property of the goal, not the measurement


def test_missing_priority_is_non_scorable():
    row = _row(priority=None)
    scored = S.score_goal(row)
    assert scored.scorable is False
    assert scored.base_score is None


# =========================================================================== #
# structure_group (§3.7)
# =========================================================================== #


def test_structure_group_target_category():
    row = _row(criteria="AtLeast", roi="PTV", structure_class="TARGET", goal_type="VolumeAtDose",
              acceptance_level=0.5, achieved_value=0.6)
    assert S.score_goal(row).structure_group == "Target"


def test_structure_group_mapped_oar():
    row = _row(roi="Kidney Lt", structure_class="OAR")
    assert S.score_goal(row).structure_group == "Kidneys"


def test_structure_group_none_for_unmapped_roi():
    row = _row(roi="BODY", structure_class="OAR")
    assert S.score_goal(row).structure_group is None


# =========================================================================== #
# compute_goal_scores — real Pt1/Pt5 pilot fixtures
# =========================================================================== #


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


def test_compute_goal_scores_has_priority_filters_required_columns(pt1_pt5_engine):
    """engine/priority_filter.py (already shipped, Module 4) declares
    exactly which columns it needs — compute_goal_scores must produce all
    of them, by name, or Module 4 breaks."""
    scores = S.compute_goal_scores(_pipeline(pt1_pt5_engine))
    assert PRIORITY_FILTER_REQUIRED_COLUMNS.issubset(set(scores.columns))


def test_compute_goal_scores_one_row_per_goal(pt1_pt5_engine):
    df = _pipeline(pt1_pt5_engine)
    scores = S.compute_goal_scores(df)
    assert len(scores) == len(df)


def test_compute_goal_scores_pt1_ptv_row(pt1_pt5_engine):
    scores = S.compute_goal_scores(_pipeline(pt1_pt5_engine))
    row = scores[(scores.patient == "Pt1") & (scores.plan == "Manual") & (scores.roi == "PTV")].iloc[0]
    assert row.category == "Target Coverage"
    assert row.scorable == True  # noqa: E712
    assert row.structure_group == "Target"
    assert row.weighted_score == pytest.approx(row.base_score * row.multiplier)
    assert row.max_weighted == pytest.approx(row.max_base * row.multiplier)


def test_compute_goal_scores_pending_goals_are_not_scorable(pt1_pt5_engine):
    """Bone Marrow (missing AchievedValue) and Rectum (missing Priority)
    stay excluded from scoring until resolved via engine/corrections.py —
    same contract as compute_pass_rate's exclusions."""
    scores = S.compute_goal_scores(_pipeline(pt1_pt5_engine))
    pt1_manual = scores[(scores.patient == "Pt1") & (scores.plan == "Manual")]
    bone = pt1_manual[pt1_manual.roi == "Bone Marrow"].iloc[0]
    rectum = pt1_manual[pt1_manual.roi == "Rectum"].iloc[0]
    assert bone.scorable == False  # noqa: E712
    assert rectum.scorable == False  # noqa: E712
    assert pd.isna(bone.weighted_score)
    assert pd.isna(rectum.weighted_score)


def test_compute_goal_scores_empty_df():
    scores = S.compute_goal_scores(pd.DataFrame())
    assert scores.empty
    assert PRIORITY_FILTER_REQUIRED_COLUMNS.issubset(set(scores.columns))


# =========================================================================== #
# compute_critical_alerts — §3.6
# =========================================================================== #


def test_critical_alert_ok_real_fixture(pt1_pt5_engine):
    """Pt1 Manual's real PTV goal: Achieved 97.2% >= 95% -> OK."""
    alerts = S.compute_critical_alerts(_pipeline(pt1_pt5_engine))
    row = alerts[(alerts.patient == "Pt1") & (alerts.plan == "Manual")].iloc[0]
    assert row.status == "OK"
    assert row.coverage == pytest.approx(0.972)
    assert row.basis is None
    assert row.target_dose_cgy == pytest.approx(4180.0)  # 95% of Pt1's 4400 cGy Rx


def test_critical_alert_no_ptv_goal_found(pt1_pt5_engine):
    """Pt1's Auto plan has no 'PTV' roi at all in this fixture (only
    'ITV') -> flagged, not silently skipped."""
    alerts = S.compute_critical_alerts(_pipeline(pt1_pt5_engine))
    row = alerts[(alerts.patient == "Pt1") & (alerts.plan == "Auto")].iloc[0]
    assert row.status == "No PTV coverage goal found"
    assert pd.isna(row.coverage)


def test_critical_alert_fail_hand_built():
    df = pd.DataFrame([dict(
        pt_no="PtX", plan_type="Manual", roi="PTV", goal_type="VolumeAtDose", criteria="AtLeast",
        parameter_value=4275.0, acceptance_level=0.95, achieved_value=0.90, rx_cgy=4500.0,
    )])
    alerts = S.compute_critical_alerts(df)
    row = alerts.iloc[0]
    assert row.status == "FAIL (Unacceptable)"
    assert row.coverage == pytest.approx(0.90)
    assert row.basis is None


def test_critical_alert_fallback_hand_built():
    """No V95%Rx goal, but a PTV goal at V100%Rx with AcceptanceLevel=95%
    -> fallback match, Basis records which dose level was actually used."""
    df = pd.DataFrame([dict(
        pt_no="PtY", plan_type="Manual", roi="PTV", goal_type="VolumeAtDose", criteria="AtLeast",
        parameter_value=4500.0, acceptance_level=0.95, achieved_value=0.93, rx_cgy=4500.0,
    )])
    alerts = S.compute_critical_alerts(df)
    row = alerts.iloc[0]
    assert row.status == "FAIL (Unacceptable)"
    assert row.basis == "fallback: V100%Rx"
    assert row.coverage == pytest.approx(0.93)


def test_critical_alert_dose_tolerance_plus_one_percent():
    """The ±1% dose tolerance: a goal 1% of Rx away from the exact 95%
    dose level still counts as the primary match."""
    rx = 4500.0
    exact_95 = 0.95 * rx  # 4275.0
    within_tolerance = exact_95 + 0.01 * rx  # +1% of Rx = +45 -> 4320.0
    df = pd.DataFrame([dict(
        pt_no="PtZ", plan_type="Manual", roi="PTV", goal_type="VolumeAtDose", criteria="AtLeast",
        parameter_value=within_tolerance, acceptance_level=0.90, achieved_value=0.96, rx_cgy=rx,
    )])
    alerts = S.compute_critical_alerts(df)
    row = alerts.iloc[0]
    assert row.status == "OK"
    assert row.basis is None  # matched as primary, not fallback


def test_critical_alert_excludes_ptv_n():
    """PTV-N (nodal boost) must never be picked up as the main-PTV check."""
    df = pd.DataFrame([dict(
        pt_no="PtN", plan_type="Manual", roi="PTV-N", goal_type="VolumeAtDose", criteria="AtLeast",
        parameter_value=4275.0, acceptance_level=0.95, achieved_value=0.99, rx_cgy=4500.0,
    )])
    alerts = S.compute_critical_alerts(df)
    row = alerts.iloc[0]
    assert row.status == "No PTV coverage goal found"


def test_critical_alert_takes_minimum_across_multiple_matches():
    """§3.6 step 5: Coverage = the minimum of the selected criteria."""
    df = pd.DataFrame([
        dict(pt_no="PtM", plan_type="Manual", roi="PTV", goal_type="VolumeAtDose", criteria="AtLeast",
            parameter_value=4275.0, acceptance_level=0.95, achieved_value=0.99, rx_cgy=4500.0),
        dict(pt_no="PtM", plan_type="Manual", roi="PTV", goal_type="VolumeAtDose", criteria="AtLeast",
            parameter_value=4275.0, acceptance_level=0.90, achieved_value=0.93, rx_cgy=4500.0),
    ])
    alerts = S.compute_critical_alerts(df)
    row = alerts.iloc[0]
    assert row.coverage == pytest.approx(0.93)  # the lower of 0.99 and 0.93
    assert row.status == "FAIL (Unacceptable)"


def test_critical_alert_empty_df():
    alerts = S.compute_critical_alerts(pd.DataFrame())
    assert alerts.empty
    assert list(alerts.columns) == ["patient", "plan", "status", "coverage", "basis",
                                     "rx_cgy", "target_dose_cgy"]


def test_critical_alert_missing_rx_falls_back_to_acceptance_level_match():
    """No rx_cgy at all -> the primary (dose-level) search can't run, but
    the fallback (AcceptanceLevel == 95%) still can."""
    df = pd.DataFrame([dict(
        pt_no="PtNoRx", plan_type="Manual", roi="PTV", goal_type="VolumeAtDose", criteria="AtLeast",
        parameter_value=4275.0, acceptance_level=0.95, achieved_value=0.80, rx_cgy=None,
    )])
    alerts = S.compute_critical_alerts(df)
    row = alerts.iloc[0]
    assert row.status == "FAIL (Unacceptable)"
    assert row.basis is not None and row.basis.startswith("fallback")
