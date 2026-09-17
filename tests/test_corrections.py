"""Tests for engine/corrections.py and the storage-layer support it needs
(GoalCorrection, replace_plan, load_analysis_frame's goal_id column)."""
from __future__ import annotations

import pandas as pd
import pytest
from sqlmodel import Session, select

from engine import corrections as C
from engine import parser as P
from engine.schemas import (
    CorrectionField,
    CorrectionSource,
    CorrectionStatus,
    CriteriaDirection,
    DoseRegimen,
    FormInput,
    GoalRow,
    GoalStatus,
    PlanFrame,
    PlanType,
    StructureClass,
)
from engine.storage import GoalCorrection, GoalResult, init_db, load_analysis_frame, replace_plan, save_case

FIXTURES = "tests/fixtures/pilot"


def _form(**overrides) -> FormInput:
    defaults = dict(
        hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4000, fractions=15,
        sib_boost=False, tx_room="Room1", mp1="Alice", mp2="Bob", ro="Dr. Carter",
        time_manual=30.0, time_auto=10.0, time_automanual=15.0,
    )
    defaults.update(overrides)
    return FormInput(**defaults)


def _goal(**overrides) -> GoalRow:
    defaults = dict(
        goal_key="k1", priority=1, roi_raw="Bone", roi="Bone Marrow",
        goal_text="V500cGy <= 850.00 cc", goal_type="AbsoluteVolumeAtDose",
        criteria=CriteriaDirection.AT_MOST, acceptance_level=850.0, parameter_value=500.0,
        achieved_value=None, status=GoalStatus.FAIL, evaluable=False,
        structure_class=StructureClass.OAR,
    )
    defaults.update(overrides)
    return GoalRow(**defaults)


@pytest.fixture
def engine_with_pending_goal():
    """A fresh in-memory DB with one patient, one Manual plan, and one goal
    row missing AchievedValue — the shape apply_corrections/record_correction
    are meant to fix."""
    engine = init_db("sqlite://")
    form = _form()
    frame = PlanFrame(plan_type=PlanType.MANUAL, planning_time_min=30.0, goals=[_goal()])
    patient_id = save_case(engine, form, [frame])
    df = load_analysis_frame(engine)
    goal_id = int(df.loc[df["roi"] == "Bone Marrow", "goal_id"].iloc[0])
    return engine, patient_id, goal_id


# --------------------------------------------------------------------------- #
# unit_for_goal_type / to_storage_value (percent -> fraction)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("goal_type,expected_unit", [
    ("DoseAtVolume", "cGy"),
    ("DoseAtAbsoluteVolume", "cGy"),
    ("AverageDose", "cGy"),
    ("AbsoluteVolumeAtDose", "cc"),
    ("VolumeAtDose", "%"),
    ("SomeFutureGoalType", "cGy"),  # falls back to the dose default
])
def test_unit_for_goal_type(goal_type, expected_unit):
    assert C.unit_for_goal_type(goal_type) == expected_unit


def test_percent_goal_type_converted_to_fraction():
    assert C.to_storage_value(67.5, "VolumeAtDose") == pytest.approx(0.675)


def test_non_percent_goal_type_passes_through_unchanged():
    assert C.to_storage_value(4820.0, "DoseAtVolume") == 4820.0
    assert C.to_storage_value(850.0, "AbsoluteVolumeAtDose") == 850.0


# --------------------------------------------------------------------------- #
# validate_value_range
# --------------------------------------------------------------------------- #


def test_validate_value_range_percent_bounds():
    C.validate_value_range(0, "VolumeAtDose")
    C.validate_value_range(100, "VolumeAtDose")
    with pytest.raises(C.CorrectionValidationError):
        C.validate_value_range(-1, "VolumeAtDose")
    with pytest.raises(C.CorrectionValidationError):
        C.validate_value_range(100.1, "VolumeAtDose")


def test_validate_value_range_dose_and_volume_nonnegative():
    C.validate_value_range(0, "DoseAtVolume")
    C.validate_value_range(4820.0, "DoseAtVolume")
    with pytest.raises(C.CorrectionValidationError):
        C.validate_value_range(-0.01, "AbsoluteVolumeAtDose")


# --------------------------------------------------------------------------- #
# is_far_from_goal
# --------------------------------------------------------------------------- #


def test_is_far_from_goal():
    assert C.is_far_from_goal(900.0, 850.0) is False       # ~6% off
    assert C.is_far_from_goal(2000.0, 850.0) is True       # >50% off
    assert C.is_far_from_goal(0.5, 0.0) is True            # zero-goal edge case
    assert C.is_far_from_goal(0.0, 0.0) is False


# --------------------------------------------------------------------------- #
# record_correction: validation
# --------------------------------------------------------------------------- #


def test_record_correction_requires_source(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    with pytest.raises(C.CorrectionValidationError):
        C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                            status=CorrectionStatus.CORRECTED, source=None,
                            reason="because", corrected_by="Dr. Test", corrected_value_display=800.0)


def test_record_correction_requires_reason(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    with pytest.raises(C.CorrectionValidationError):
        C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                            status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                            reason="   ", corrected_by="Dr. Test", corrected_value_display=800.0)


def test_record_correction_requires_value_when_status_is_corrected(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    with pytest.raises(C.CorrectionValidationError):
        C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                            status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                            reason="because", corrected_by="Dr. Test", corrected_value_display=None)


def test_record_correction_rejects_confirmed_not_evaluable_for_priority(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    with pytest.raises(C.CorrectionValidationError):
        C.record_correction(engine, goal_id=goal_id, field=CorrectionField.PRIORITY,
                            status=CorrectionStatus.CONFIRMED_NOT_EVALUABLE, source=CorrectionSource.PROTOCOL,
                            reason="because", corrected_by="Dr. Test")


def test_record_correction_rejects_invalid_priority(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    with pytest.raises(C.CorrectionValidationError):
        C.record_correction(engine, goal_id=goal_id, field=CorrectionField.PRIORITY,
                            status=CorrectionStatus.CORRECTED, source=CorrectionSource.PROTOCOL,
                            reason="because", corrected_by="Dr. Test", corrected_value_display=5)


def test_record_correction_rejects_out_of_range_value(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    with pytest.raises(C.CorrectionValidationError):
        C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                            status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                            reason="because", corrected_by="Dr. Test", corrected_value_display=-5.0)


def test_record_correction_warns_but_saves_when_far_from_goal(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    correction_id, warning = C.record_correction(
        engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
        reason="way off", corrected_by="Dr. Test", corrected_value_display=2000.0,
    )
    assert correction_id is not None
    assert warning is not None and "more than 50%" in warning


def test_record_correction_no_warning_when_close_to_goal(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    _, warning = C.record_correction(
        engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
        reason="close", corrected_by="Dr. Test", corrected_value_display=820.0,
    )
    assert warning is None


# --------------------------------------------------------------------------- #
# record_correction + apply_corrections: applied correctly
# --------------------------------------------------------------------------- #


def test_correction_applied_correctly_recomputes_status(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.RAYSTATION_DVH,
                        reason="measured", corrected_by="Dr. Test", corrected_value_display=820.0)

    corrected = C.apply_corrections(load_analysis_frame(engine), engine)
    row = corrected[corrected["goal_id"] == goal_id].iloc[0]

    assert row["achieved_value"] == 820.0
    assert row["status"] == "PASS"  # AtMost: 820 <= 850
    assert row["evaluable"] == True  # noqa: E712
    assert row["correction_tag"] == "Corrected"
    assert row["is_pending_review"] == False  # noqa: E712


def test_correction_applied_correctly_fail_case(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.RAYSTATION_DVH,
                        reason="measured", corrected_by="Dr. Test", corrected_value_display=900.0)

    corrected = C.apply_corrections(load_analysis_frame(engine), engine)
    row = corrected[corrected["goal_id"] == goal_id].iloc[0]
    assert row["status"] == "FAIL"  # AtMost: 900 > 850


def test_priority_correction_applied_correctly(engine_with_pending_goal):
    engine, _, _ = engine_with_pending_goal
    df = load_analysis_frame(engine)
    goal_id = int(df.loc[df["roi"] == "Bone Marrow", "goal_id"].iloc[0])
    # this fixture's goal already has priority=1; force it missing to test resolution
    with Session(engine) as s:
        g = s.get(GoalResult, goal_id)
        g.priority = None
        s.add(g)
        s.commit()

    C.record_correction(engine, goal_id=goal_id, field=CorrectionField.PRIORITY,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.PROTOCOL,
                        reason="per protocol table", corrected_by="Dr. Test", corrected_value_display=2)

    corrected = C.apply_corrections(load_analysis_frame(engine), engine)
    row = corrected[corrected["goal_id"] == goal_id].iloc[0]
    assert row["priority"] == 2
    assert row["correction_tag"] == "Corrected"


def test_confirmed_not_evaluable_resolves_pending_without_a_value(engine_with_pending_goal):
    engine, _, goal_id = engine_with_pending_goal
    C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CONFIRMED_NOT_EVALUABLE, source=CorrectionSource.OTHER,
                        reason="not contoured on this plan", corrected_by="Dr. Test")

    corrected = C.apply_corrections(load_analysis_frame(engine), engine)
    row = corrected[corrected["goal_id"] == goal_id].iloc[0]

    assert pd.isna(row["achieved_value"])
    assert row["evaluable"] == False  # noqa: E712
    assert row["status"] is None or pd.isna(row["status"])
    assert row["confirmed_not_evaluable"] == True  # noqa: E712
    assert row["correction_tag"] == "Confirmed not evaluable"
    assert row["is_pending_review"] == False  # noqa: E712


def test_percent_conversion_end_to_end():
    """A VolumeAtDose goal's AchievedValue is stored as a fraction (0-1);
    entering 67.5 (%) must save as 0.675, not 67.5."""
    engine = init_db("sqlite://")
    goal = _goal(goal_key="k2", roi="PTV45", goal_type="VolumeAtDose",
                criteria=CriteriaDirection.AT_LEAST, acceptance_level=0.95, parameter_value=95.0,
                achieved_value=None, status=None, evaluable=False, structure_class=StructureClass.TARGET)
    frame = PlanFrame(plan_type=PlanType.MANUAL, goals=[goal])
    save_case(engine, _form(pt_no="Pt2", hn="90000002"), [frame])

    df = load_analysis_frame(engine)
    goal_id = int(df.loc[df["roi"] == "PTV45", "goal_id"].iloc[0])

    C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.PLAN_REPORT,
                        reason="from plan report", corrected_by="Dr. Test", corrected_value_display=97.2)

    corrected = C.apply_corrections(load_analysis_frame(engine), engine)
    row = corrected[corrected["goal_id"] == goal_id].iloc[0]
    assert row["achieved_value"] == pytest.approx(0.972)
    assert row["status"] == "PASS"  # AtLeast: 0.972 >= 0.95


# --------------------------------------------------------------------------- #
# Re-upload supersedes a correction
# --------------------------------------------------------------------------- #


def test_reupload_supersedes_a_correction(engine_with_pending_goal):
    engine, patient_id, goal_id = engine_with_pending_goal
    C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                        reason="first correction", corrected_by="Dr. Test", corrected_value_display=800.0)

    with Session(engine) as s:
        assert len(s.exec(select(GoalCorrection)
                          .where(GoalCorrection.superseded_by_upload == False)).all()) == 1  # noqa: E712

    new_frame = PlanFrame(plan_type=PlanType.MANUAL, planning_time_min=30.0, goals=[_goal()])
    replace_plan(engine, patient_id, new_frame)

    with Session(engine) as s:
        all_corrections = s.exec(select(GoalCorrection)).all()
        assert len(all_corrections) == 1  # kept, not deleted
        assert all_corrections[0].superseded_by_upload is True

    corrected = C.apply_corrections(load_analysis_frame(engine), engine)
    # the re-uploaded row is missing AchievedValue again — no active correction applies
    assert corrected["is_pending_review"].any()
    assert not (corrected["correction_tag"] == "Corrected").any()


def test_reupload_of_new_plan_type_does_not_affect_other_plans(engine_with_pending_goal):
    engine, patient_id, goal_id = engine_with_pending_goal
    C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                        reason="manual plan correction", corrected_by="Dr. Test",
                        corrected_value_display=820.0)

    # uploading an *Auto* plan for the same patient must not touch the Manual correction
    replace_plan(engine, patient_id, PlanFrame(plan_type=PlanType.AUTO, goals=[_goal(goal_key="k3")]))

    with Session(engine) as s:
        assert s.get(GoalCorrection, 1).superseded_by_upload is False

    corrected = C.apply_corrections(load_analysis_frame(engine), engine)
    manual_row = corrected[(corrected["plan_type"] == "Manual") & (corrected["goal_id"] == goal_id)].iloc[0]
    assert manual_row["achieved_value"] == 820.0


# --------------------------------------------------------------------------- #
# Straight-pass copies corrections along with the row
# --------------------------------------------------------------------------- #


def test_straight_pass_copies_correction_to_new_goal(engine_with_pending_goal):
    """Simulates what engine/imputation.py's straight-pass rule will do:
    duplicate a goal_results row into another plan. copy_corrections_to_new_goal
    is the half of that concerned with not leaving the correction behind."""
    engine, _, source_goal_id = engine_with_pending_goal
    C.record_correction(engine, goal_id=source_goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                        reason="measured", corrected_by="Dr. Test", corrected_value_display=800.0)

    with Session(engine) as s:
        source = s.get(GoalResult, source_goal_id)
        copy = GoalResult(
            plan_id=source.plan_id, goal_key=source.goal_key, priority=source.priority,
            roi_raw=source.roi_raw, roi=source.roi, goal_text=source.goal_text,
            goal_type=source.goal_type, criteria=source.criteria,
            acceptance_level=source.acceptance_level, parameter_value=source.parameter_value,
            achieved_value=source.achieved_value, status=source.status,
            evaluable=source.evaluable, structure_class=source.structure_class,
        )
        s.add(copy)
        s.commit()
        s.refresh(copy)
        target_goal_id = copy.id

    n_copied = C.copy_corrections_to_new_goal(engine, source_goal_id=source_goal_id,
                                              target_goal_id=target_goal_id)
    assert n_copied == 1

    with Session(engine) as s:
        target_corrections = s.exec(
            select(GoalCorrection).where(GoalCorrection.goal_id == target_goal_id)
        ).all()
    assert len(target_corrections) == 1
    assert target_corrections[0].corrected_value == "800.0"
    assert "copied by straight-pass" in target_corrections[0].reason


def test_straight_pass_copy_skips_superseded_corrections(engine_with_pending_goal):
    engine, patient_id, source_goal_id = engine_with_pending_goal
    C.record_correction(engine, goal_id=source_goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                        reason="will be superseded", corrected_by="Dr. Test",
                        corrected_value_display=800.0)
    replace_plan(engine, patient_id, PlanFrame(plan_type=PlanType.MANUAL, goals=[_goal()]))  # supersedes it

    n_copied = C.copy_corrections_to_new_goal(engine, source_goal_id=source_goal_id, target_goal_id=999)
    assert n_copied == 0


# --------------------------------------------------------------------------- #
# Pending goals stay excluded from scoring/pass rate until resolved
# --------------------------------------------------------------------------- #


def test_pending_goals_excluded_until_resolved():
    """The contract apply_corrections provides for scoring/pass-rate (built
    in a later phase): a pending row has Priority NaN and/or Status not
    set, so filtering on those naturally excludes it — until a correction
    resolves it, at which point is_pending_review flips to False."""
    engine = init_db("sqlite://")
    goals = [
        _goal(goal_key="missing_achieved", roi="Bone Marrow"),  # achieved_value=None
        _goal(goal_key="missing_priority", roi="Sigmoid", priority=None,
              achieved_value=3200.0, status=GoalStatus.PASS, evaluable=True),
        _goal(goal_key="fully_resolved", roi="Rectum", achieved_value=2800.0,
              status=GoalStatus.PASS, evaluable=True),
    ]
    frame = PlanFrame(plan_type=PlanType.MANUAL, goals=goals)
    save_case(engine, _form(), [frame])

    corrected = C.apply_corrections(load_analysis_frame(engine), engine)
    pending = corrected[corrected["is_pending_review"]]
    assert set(pending["roi"]) == {"Bone Marrow", "Sigmoid"}

    # a naive scoring/pass-rate filter would only see fully-resolved + priority-set rows
    scoreable = corrected[corrected["priority"].notna() & corrected["status"].notna()]
    assert set(scoreable["roi"]) == {"Rectum"}

    # resolve both pending goals
    bone_id = int(corrected.loc[corrected["roi"] == "Bone Marrow", "goal_id"].iloc[0])
    sigmoid_id = int(corrected.loc[corrected["roi"] == "Sigmoid", "goal_id"].iloc[0])
    C.record_correction(engine, goal_id=bone_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                        reason="measured", corrected_by="Dr. Test", corrected_value_display=800.0)
    C.record_correction(engine, goal_id=sigmoid_id, field=CorrectionField.PRIORITY,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.PROTOCOL,
                        reason="per protocol", corrected_by="Dr. Test", corrected_value_display=2)

    resolved = C.apply_corrections(load_analysis_frame(engine), engine)
    assert not resolved["is_pending_review"].any()
    scoreable_after = resolved[resolved["priority"].notna() & resolved["status"].notna()]
    assert set(scoreable_after["roi"]) == {"Bone Marrow", "Sigmoid", "Rectum"}
